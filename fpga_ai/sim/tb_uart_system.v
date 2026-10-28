`timescale 1ns/1ps
module tb_uart_system;
    parameter CPB=8;
    localparam TIMEOUT=CPB*120;
    reg clk=0,reset=1,rx=1;
    wire tx,dp;
    wire [6:0] segments;
    wire [3:0] an;
    digit_ai_top #(.CLOCK_HZ(CPB*115200),.BAUD(115200),.PACKET_TIMEOUT_CYCLES(TIMEOUT)) dut(clk,reset,rx,tx,segments,an,dp);
    always #5 clk=~clk;
    reg [7:0] pixels[0:11759];
    reg signed [31:0] expected_trace[0:3029];
    reg [7:0] expected_digit[0:14],expected_unknown[0:14];
    reg [31:0] expected_margin[0:14];
    integer ack[0:255],result[0:255],error_status[0:255];
    integer byte_count=0,responses=0,result_checks=0,errors_checked=0;
    reg [7:0] response[0:60];
    reg [7:0] received_byte;
    reg [15:0] response_crc,send_crc;
    reg signed [31:0] received_score;
    integer sample,i,b,j,sequence_value,status_value;
    reg [15:0] held_display;
    function [15:0] crc_byte;
        input [15:0] old_crc; input [7:0] value;
        reg [15:0] c;integer k;
        begin c=old_crc^{value,8'd0};for(k=0;k<8;k=k+1)c=c[15]?(c<<1)^16'h1021:c<<1;crc_byte=c;end
    endfunction
    task send_byte;
        input [7:0] value;
        integer bit_number;
        begin
            @(negedge clk);rx=0;repeat(CPB) @(negedge clk);
            for(bit_number=0;bit_number<8;bit_number=bit_number+1) begin rx=value[bit_number];repeat(CPB) @(negedge clk);end
            rx=1;repeat(CPB) @(negedge clk);
        end
    endtask
    task body_byte;
        input [7:0] value;
        begin send_crc=crc_byte(send_crc,value);send_byte(value);end
    endtask
    task header;
        input [7:0] seq;
        input [15:0] length;
        begin
            send_crc=16'hffff;send_byte(8'ha5);send_byte(8'h5a);
            body_byte(1);body_byte(1);body_byte(seq);body_byte(length[7:0]);body_byte(length[15:8]);
        end
    endtask
    task send_image;
        input integer image_index;
        input [7:0] seq;
        input corrupt;
        integer p;
        begin
            header(seq,784);
            for(p=0;p<784;p=p+1) body_byte(pixels[image_index*784+p]);
            send_byte(send_crc[7:0] ^ (corrupt?8'h01:8'h00));send_byte(send_crc[15:8]);
        end
    endtask
    task wait_result;
        input integer seq;
        begin wait(result[seq]==1);repeat(5) @(negedge clk);end
    endtask
    task check_error;
        input integer seq,expected_status;
        begin
            wait(error_status[seq]!=0);
            if(error_status[seq]!=expected_status) $fatal(1,"Wrong error seq%0d got%0d expected%0d",seq,error_status[seq],expected_status);
            if(dut.display_bcd!==held_display) $fatal(1,"Invalid packet changed display");
            errors_checked=errors_checked+1;
        end
    endtask

    // Independent UART decoder (not the DUT receiver) and packet CRC checker.
    initial forever begin
        @(negedge tx); #(CPB*15);
        for(b=0;b<8;b=b+1) begin received_byte[b]=tx;#(CPB*10);end
        if(tx!==1) $fatal(1,"Bad FPGA TX stop bit");
        response[byte_count]=received_byte;byte_count=byte_count+1;
        if(byte_count==61) begin
            byte_count=0;responses=responses+1;
            if(response[0]!==8'ha5 || response[1]!==8'h5a || response[2]!==1 || response[3]!==8'h81 || response[5]!==52 || response[6]!==0)
                $fatal(1,"Bad response header");
            response_crc=16'hffff;
            for(j=2;j<59;j=j+1) response_crc=crc_byte(response_crc,response[j]);
            if({response[60],response[59]}!==response_crc) $fatal(1,"Bad response CRC");
            sequence_value=response[4];status_value=response[7];
            if(status_value==1) ack[sequence_value]=ack[sequence_value]+1;
            else if(status_value==0) begin
                if(sequence_value>14) $fatal(1,"Unexpected inference result seq%0d",sequence_value);
                if(ack[sequence_value]!=1) $fatal(1,"Result without exactly one ACK");
                if(response[8]!==expected_digit[sequence_value] || response[9]!==expected_unknown[sequence_value]) $fatal(1,"Decision response mismatch");
                if({response[14],response[13],response[12],response[11]}!==32'd328190) $fatal(1,"Response latency mismatch");
                if({response[18],response[17],response[16],response[15]}!==expected_margin[sequence_value]) $fatal(1,"Response margin mismatch");
                for(j=0;j<10;j=j+1) begin
                    received_score={response[22+j*4],response[21+j*4],response[20+j*4],response[19+j*4]};
                    if(received_score!==expected_trace[sequence_value*202+192+j]) $fatal(1,"UART logit mismatch seq%0d output%0d",sequence_value,j);
                end
                result[sequence_value]=result[sequence_value]+1;result_checks=result_checks+1;
                $display("UART_IMAGE_PASS seq=%0d digit=%0d unknown=%0d",sequence_value,response[8],response[9]);
            end else error_status[sequence_value]=status_value;
        end
    end
    initial begin
        $readmemh("../sim/data/pixels.mem",pixels);$readmemh("../sim/data/trace.mem",expected_trace);
        $readmemh("../sim/data/digits.mem",expected_digit);$readmemh("../sim/data/unknowns.mem",expected_unknown);
        $readmemh("../sim/data/margins.mem",expected_margin);
        for(i=0;i<256;i=i+1) begin ack[i]=0;result[i]=0;error_status[i]=0;end
        repeat(8) @(negedge clk);reset=0;repeat(8) @(negedge clk);
        // Full request while inference is busy: drain it, reject it, and never
        // overwrite the active image. Original seq0 must still match exactly.
        send_image(0,0,0);wait(ack[0]==1);send_image(1,32,0);
        wait_result(0);held_display=16'h0007;check_error(32,5);
        for(sample=1;sample<15;sample=sample+1) begin
            send_image(sample,sample,0);wait_result(sample);
            if(dut.display_bcd!==(expected_unknown[sample] ? 16'h9999 : {12'd0,expected_digit[sample][3:0]})) $fatal(1,"Display BCD mismatch");
        end
        held_display=dut.display_bcd;
        send_image(0,40,1);check_error(40,2);
        header(41,784);body_byte(8'hff);check_error(41,4);
        header(42,783);check_error(42,3);
        // Framing error during a packet: low stop bit must not be accepted.
        header(43,784);
        @(negedge clk);rx=0;repeat(CPB*10) @(negedge clk);rx=1;
        check_error(43,7);
        if(result_checks!=15 || errors_checked!=5) $fatal(1,"Missing system checks");
        $display("PASS: UART end-to-end 15 images / 150 exact logits, ACK+CRC, busy/CRC/timeout/length/framing errors, display hold; CPB=%0d",CPB);
        $finish;
    end
    initial begin #2000000000;$fatal(1,"UART system watchdog");end
endmodule
