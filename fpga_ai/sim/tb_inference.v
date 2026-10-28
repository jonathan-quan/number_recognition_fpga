`timescale 1ns/1ps
module tb_inference;
    reg clk=0,reset=1,start=0,pixel_write=0,image_blank=0;
    reg [9:0] pixel_address=0;
    reg [7:0] pixel_data=0;
    wire busy,done,unknown,trace_valid;
    wire [3:0] digit;
    wire [31:0] margin,cycles;
    wire signed [31:0] score_data,trace_value;
    wire [1:0] trace_layer;
    wire [7:0] trace_neuron;
    reg [3:0] score_address=0;
    ai_inference dut(clk,reset,start,pixel_write,pixel_address,pixel_data,image_blank,
        busy,done,digit,unknown,margin,cycles,score_address,score_data,
        trace_valid,trace_layer,trace_neuron,trace_value);
    always #5 clk=~clk;
    reg [7:0] pixels[0:11759];
    reg [7:0] normalized[0:11759];
    reg signed [31:0] expected_trace[0:3029],expected_acc[0:3029];
    reg [7:0] expected_digit[0:14],expected_unknown[0:14];
    reg [31:0] expected_margin[0:14];
    reg signed [31:0] confidence_scores[0:399];
    reg [31:0] confidence_sums[0:39];
    reg [7:0] confidence_unknown[0:39];
    integer confidence_case,j;
    integer sample=0,i,offset,checks=0,trace_checks=0;
    reg signed [31:0] forced_acc;
    reg [1:0] forced_layer;
    integer qvalue,round_checks=0,k,l;
    always @(posedge clk) if(!reset && trace_valid) begin
        offset=sample*202+(trace_layer==0 ? 0 : trace_layer==1 ? 128 : 192)+trace_neuron;
        if(trace_value!==expected_trace[offset]) $fatal(1,"Output mismatch sample%0d layer%0d neuron%0d got%0d expected%0d",sample,trace_layer,trace_neuron,trace_value,expected_trace[offset]);
        if(dut.accumulator!==expected_acc[offset]) $fatal(1,"Accumulator mismatch sample%0d offset%0d",sample,offset);
        trace_checks=trace_checks+1;
    end
    initial begin
        $readmemh("../sim/data/pixels.mem",pixels);
        $readmemh("../sim/data/normalized.mem",normalized);
        $readmemh("../sim/data/trace.mem",expected_trace);
        $readmemh("../sim/data/accumulators.mem",expected_acc);
        $readmemh("../sim/data/digits.mem",expected_digit);
        $readmemh("../sim/data/unknowns.mem",expected_unknown);
        $readmemh("../sim/data/margins.mem",expected_margin);
        $readmemh("../sim/data/confidence_scores.mem",confidence_scores);
        $readmemh("../sim/data/confidence_sums.mem",confidence_sums);
        $readmemh("../sim/data/confidence_unknown.mem",confidence_unknown);
        // Sweep the rounding/saturation boundary region for both shifts,
        // independently of the reference-generated image vectors.
        force dut.accumulator=forced_acc;
        force dut.layer=forced_layer;
        for(l=0;l<2;l=l+1) begin
            forced_layer=l;
            for(k=-1024;k<=132096;k=k+1) begin
                forced_acc=k;#1;
                qvalue=k<=0 ? 0 : (k+(l==0?512:128))>>(l==0?10:8);
                if(qvalue>127) qvalue=127;
                if(dut.requantized!==qvalue[7:0]) $fatal(1,"Rounding boundary mismatch layer%0d acc%0d",l,k);
                round_checks=round_checks+1;
            end
            forced_acc=32'sh7fffffff;#1;if(dut.requantized!==127) $fatal(1,"Positive overflow in rounding addition");
            forced_acc=32'sh80000000;#1;if(dut.requantized!==0) $fatal(1,"Negative ReLU boundary failed");
        end
        release dut.accumulator;release dut.layer;
        repeat(4) @(negedge clk);reset=0;
        for(sample=0;sample<15;sample=sample+1) begin
            for(i=0;i<784;i=i+1) begin
                @(negedge clk);pixel_write=1;pixel_address=i;pixel_data=pixels[sample*784+i];
            end
            @(negedge clk);pixel_write=0;
            for(i=0;i<784;i=i+1)
                if(dut.image[i]!==normalized[sample*784+i]) $fatal(1,"Input LUT/buffer mismatch");
            image_blank=sample==10;start=1;
            @(negedge clk);start=0;
            wait(busy); // A start while busy must be ignored.
            repeat(5) @(negedge clk);start=1;
            @(negedge clk);start=0;
            wait(done);@(negedge clk);
            if(digit!==expected_digit[sample][3:0] || unknown!==expected_unknown[sample][0] || margin!==expected_margin[sample])
                $fatal(1,"Decision mismatch sample%0d digit%0d unknown%0d margin%0d",sample,digit,unknown,margin);
            if(cycles!==328190) $fatal(1,"Latency mismatch %0d",cycles);
            for(i=0;i<10;i=i+1) begin
                score_address=i;#1;
                if(score_data!==expected_trace[sample*202+192+i]) $fatal(1,"Score read mismatch");
            end
            checks=checks+1;
            $display("CORE_PASS sample=%0d digit=%0d unknown=%0d cycles=%0d",sample,digit,unknown,cycles);
        end
        if(trace_checks!=3030) $fatal(1,"Missing trace values %0d",trace_checks);
        $display("PASS: inference %0d images, %0d layer outputs and %0d accumulators exact; 11760 normalized pixels exact",checks,trace_checks,trace_checks);
        $display("PASS: requantization boundary sweep %0d values plus signed INT32 extremes",round_checks);
        // Inject scores after the MAC pipeline to isolate the hardware rejection
        // controller: ties, high competitor counts, bin edges and INT32 tails.
        for(confidence_case=0;confidence_case<40;confidence_case=confidence_case+1) begin
            @(negedge clk);
            for(j=0;j<10;j=j+1) dut.scores[j]=confidence_scores[confidence_case*10+j];
            dut.blank=0;dut.cycle_count=0;dut.state=6; // ARG_INIT
            wait(busy);wait(done);@(negedge clk);
            if(dut.exponential_sum!==confidence_sums[confidence_case] || unknown!==confidence_unknown[confidence_case][0])
                $fatal(1,"Confidence rejection mismatch case%0d sum%0d",confidence_case,dut.exponential_sum);
        end
        $display("PASS: 40 isolated FPGA confidence cases, bin boundaries, tails, ties and signed extremes");
        $finish;
    end
    initial begin #100000000;$fatal(1,"Inference watchdog");end
endmodule
