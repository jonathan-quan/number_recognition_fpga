`timescale 1ns/1ps
module tb_components;
    parameter CPB=8;
    reg clk=0,reset=1,load_bias=0,enable=0;
    reg signed [31:0] bias=0;
    reg signed [7:0] a=0,b=0;
    wire signed [31:0] accumulator;
    ai_mac mac(clk,reset,load_bias,enable,bias,a,b,accumulator);
    always #5 clk=~clk;
    reg tx_start=0;
    reg [7:0] tx_data=0;
    wire tx,tx_busy;
    reg inject=0,manual_rx=1;
    wire [7:0] rx_data;
    wire rx_valid,rx_error;
    ai_uart_tx #(.CLOCK_HZ(CPB*115200),.BAUD(115200)) utx(clk,reset,tx_start,tx_data,tx,tx_busy);
    ai_uart_rx #(.CLOCK_HZ(CPB*115200),.BAUD(115200)) urx(clk,reset,inject?manual_rx:tx,rx_data,rx_valid,rx_error);
    reg [15:0] bcd=0;
    wire [6:0] segments;
    wire [3:0] an;
    wire dp;
    ai_display display_unit(clk,reset,bcd,segments,an,dp);
    reg [1:0] phase=0;
    reg [6:0] segment_table[0:9];
    integer value,bit_number,tick,seen=0,framing=0,checks=0,d,p;
    reg expected_bit;
    always @(posedge clk) begin
        if(rx_valid) begin
            if(rx_data!==seen[7:0]) $fatal(1,"UART RX byte mismatch got%0d expected%0d",rx_data,seen);
            seen=seen+1;
        end
        if(rx_error) framing=framing+1;
    end
    task mac_step;
        input signed [7:0] x,y;
        input signed [31:0] expected;
        begin
            @(negedge clk);a=x;b=y;enable=1;
            @(negedge clk);enable=0;
            if(accumulator!==expected) $fatal(1,"MAC signed arithmetic mismatch got%0d expected%0d",accumulator,expected);
            checks=checks+1;
        end
    endtask
    initial begin
        segment_table[0]=7'b0000001;segment_table[1]=7'b1001111;segment_table[2]=7'b0010010;
        segment_table[3]=7'b0000110;segment_table[4]=7'b1001100;segment_table[5]=7'b0100100;
        segment_table[6]=7'b0100000;segment_table[7]=7'b0001111;segment_table[8]=7'b0000000;segment_table[9]=7'b0000100;
        repeat(4) @(negedge clk);reset=0;bias=-12345;load_bias=1;
        @(negedge clk);load_bias=0;
        if(accumulator!==-12345) $fatal(1,"Bias load failed");
        mac_step(-128,-128,4039);mac_step(-128,127,-12217);mac_step(127,127,3912);mac_step(-1,1,3911);
        repeat(3) @(negedge clk);if(accumulator!==3911) $fatal(1,"MAC enable hold failed");
        for(d=0;d<10;d=d+1) begin
            bcd={12'd0,d[3:0]};
            for(p=0;p<4;p=p+1) begin
                phase=p;force display_unit.refresh={phase,15'd0};#1;
                if(an!==(4'b1111^(4'b0001<<p)) || segments!==segment_table[p==0?d:0] || dp!==1)
                    $fatal(1,"Seven-segment 000d mismatch digit%0d phase%0d",d,p);
                checks=checks+1;
            end
        end
        bcd=16'h9999;
        for(p=0;p<4;p=p+1) begin phase=p;force display_unit.refresh={phase,15'd0};#1;if(segments!==segment_table[9]) $fatal(1,"9999 decode failed");checks=checks+1;end
        release display_unit.refresh;
        for(value=0;value<256;value=value+1) begin
            @(negedge clk);tx_data=value;tx_start=1;
            @(posedge clk);#1;
            for(bit_number=0;bit_number<10;bit_number=bit_number+1) begin
                expected_bit=bit_number==0?0:bit_number==9?1:((value>>(bit_number-1))&1);
                for(tick=0;tick<CPB;tick=tick+1) begin
                    if(tx!==expected_bit || tx_busy!==1) $fatal(1,"TX bit or bit duration mismatch");
                    checks=checks+1;
                    @(negedge clk);tx_start=bit_number==3;tx_data=~value;
                    @(posedge clk);#1;
                end
            end
            tx_start=0;
            if(tx_busy!==0 || tx!==1) $fatal(1,"TX stop completion mismatch");
            repeat(3) @(negedge clk);
        end
        if(seen!=256 || framing!=0) $fatal(1,"UART loopback count/error mismatch");
        // Less than half a bit low is a false start, not a byte.
        inject=1;manual_rx=0;repeat(CPB/4) @(negedge clk);manual_rx=1;
        repeat(CPB*12) @(negedge clk);
        if(seen!=256) $fatal(1,"False start produced byte");
        // A full low frame has an invalid stop bit.
        manual_rx=0;repeat(CPB*10) @(negedge clk);manual_rx=1;
        repeat(CPB*3) @(negedge clk);
        if(framing!=1 || seen!=256) $fatal(1,"Framing detection failed");
        // Reset must release TX and abort an in-progress character.
        inject=0;tx_start=1;@(negedge clk);tx_start=0;reset=1;
        @(negedge clk);if(tx!==1 || tx_busy!==0 || accumulator!==0) $fatal(1,"Reset abort failed");
        $display("PASS: component checks=%0d; UART all256 bytes, busy writes, bit timing, false start, framing and reset; signed MAC, bias/hold; 44 display combinations; CPB=%0d",checks,CPB);
        $finish;
    end
    initial begin #100000000;$fatal(1,"Components watchdog");end
endmodule
