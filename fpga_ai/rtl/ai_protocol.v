// Request: A5 5A 01 01 seq 10 03 [784 raw pixels] crc_lo crc_hi.
// CRC16/CCITT-FALSE over version through payload (poly1021, initFFFF).
// Response: same framing, type81, payload52: status,digit,flags,reserved,
// cycles(u32le),margin(u32le),10 signed i32le logits. See PROTOCOL.md.
module ai_protocol #(parameter TIMEOUT_CYCLES=10000000) (
    input wire clk, reset,
    input wire rx_valid, rx_error,
    input wire [7:0] rx_data,
    output wire tx_start,
    output reg [7:0] tx_data,
    input wire tx_busy,
    input wire core_busy, core_done,
    input wire [3:0] core_digit,
    input wire core_unknown,
    input wire [31:0] core_cycles, core_margin,
    output reg [3:0] score_address,
    input wire signed [31:0] score_data,
    output reg core_start,
    output reg pixel_write,
    output reg [9:0] pixel_address,
    output reg [7:0] pixel_data,
    output reg image_blank
);
    function [15:0] crc_byte;
        input [15:0] old_crc;
        input [7:0] byte_value;
        reg [15:0] value;
        integer k;
        begin
            value=old_crc ^ {byte_value,8'd0};
            for(k=0;k<8;k=k+1) value=value[15] ? (value<<1)^16'h1021 : value<<1;
            crc_byte=value;
        end
    endfunction
    localparam SYNC0=0,SYNC1=1,VERSION=2,TYPE=3,SEQ=4,LEN0=5,LEN1=6,
               PAYLOAD=7,CRC0=8,CRC1=9;
    reg [3:0] rx_state;
    reg [7:0] version,kind,sequence_id,length_low,crc_low;
    reg [15:0] rx_crc;
    reg [9:0] received;
    reg any_ink,blocked;
    reg [31:0] timeout_count;
    reg event_valid;
    reg [7:0] event_status,event_sequence;
    reg [7:0] fifo_status [0:3];
    reg [7:0] fifo_sequence [0:3];
    reg [1:0] write_pointer,read_pointer;
    reg [2:0] fifo_count;
    reg result_pending;
    reg [7:0] result_sequence,active_sequence;
    reg [2:0] tx_state;
    reg [5:0] tx_index;
    reg [7:0] status,tx_sequence;
    reg [15:0] tx_crc;
    wire pop_fifo=tx_state==0 && fifo_count!=0;
    wire push_fifo=event_valid && fifo_count<4;
    wire take_result=tx_state==0 && fifo_count==0 && result_pending;
    wire occupied=core_busy || core_start || result_pending || tx_state!=0 || fifo_count!=0 || event_valid;

    always @(posedge clk) begin
        if(reset) begin
            write_pointer<=0; read_pointer<=0; fifo_count<=0;
        end else begin
            if(push_fifo) begin
                fifo_status[write_pointer]<=event_status;
                fifo_sequence[write_pointer]<=event_sequence;
                write_pointer<=write_pointer+1'b1;
            end
            if(pop_fifo) read_pointer<=read_pointer+1'b1;
            case({push_fifo,pop_fifo})
                2'b10: fifo_count<=fifo_count+1'b1;
                2'b01: fifo_count<=fifo_count-1'b1;
                default: begin end
            endcase
        end
    end
    always @(posedge clk) begin
        if(reset) begin
            result_pending<=0; result_sequence<=0; active_sequence<=0;
        end else begin
            if(core_start) active_sequence<=sequence_id;
            if(core_done) begin result_pending<=1; result_sequence<=active_sequence; end
            else if(take_result) result_pending<=0;
        end
    end

    always @(posedge clk) begin
        if(reset) begin
            rx_state<=SYNC0; version<=0; kind<=0; sequence_id<=0;
            length_low<=0; crc_low<=0; rx_crc<=16'hffff; received<=0;
            any_ink<=0; blocked<=0; timeout_count<=0; event_valid<=0;
            event_status<=0; event_sequence<=0; core_start<=0;
            pixel_write<=0; pixel_address<=0; pixel_data<=0; image_blank<=1;
        end else begin
            event_valid<=0; core_start<=0; pixel_write<=0;
            if(rx_valid) timeout_count<=0;
            else if(rx_state!=SYNC0) timeout_count<=timeout_count+1'b1;
            else timeout_count<=0;
            if(rx_error) begin
                rx_state<=SYNC0; event_valid<=1; event_status<=7; event_sequence<=sequence_id;
            end else if(!rx_valid && rx_state!=SYNC0 && timeout_count>=TIMEOUT_CYCLES-1) begin
                rx_state<=SYNC0; event_valid<=1; event_status<=4; event_sequence<=sequence_id;
            end else if(rx_valid) begin
                case(rx_state)
                    SYNC0: if(rx_data==8'ha5) rx_state<=SYNC1;
                    SYNC1: if(rx_data==8'h5a) begin
                               rx_state<=VERSION; rx_crc<=16'hffff; blocked<=occupied; sequence_id<=0;
                           end else if(rx_data!=8'ha5) rx_state<=SYNC0;
                    VERSION: begin version<=rx_data; rx_crc<=crc_byte(rx_crc,rx_data); rx_state<=TYPE; end
                    TYPE: begin kind<=rx_data; rx_crc<=crc_byte(rx_crc,rx_data); rx_state<=SEQ; end
                    SEQ: begin sequence_id<=rx_data; rx_crc<=crc_byte(rx_crc,rx_data); rx_state<=LEN0; end
                    LEN0: begin length_low<=rx_data; rx_crc<=crc_byte(rx_crc,rx_data); rx_state<=LEN1; end
                    LEN1: begin
                        rx_crc<=crc_byte(rx_crc,rx_data);
                        if(version!=1 || kind!=1) begin
                            event_valid<=1; event_status<=6; event_sequence<=sequence_id; rx_state<=SYNC0;
                        end else if({rx_data,length_low}!=16'd784) begin
                            event_valid<=1; event_status<=3; event_sequence<=sequence_id; rx_state<=SYNC0;
                        end else begin
                            blocked<=blocked || occupied; received<=0; any_ink<=0; rx_state<=PAYLOAD;
                        end
                    end
                    PAYLOAD: begin
                        rx_crc<=crc_byte(rx_crc,rx_data);
                        any_ink<=any_ink || rx_data!=0;
                        if(!blocked) begin pixel_write<=1; pixel_address<=received; pixel_data<=rx_data; end
                        if(received==783) rx_state<=CRC0; else received<=received+1'b1;
                    end
                    CRC0: begin crc_low<=rx_data; rx_state<=CRC1; end
                    CRC1: begin
                        rx_state<=SYNC0; event_valid<=1; event_sequence<=sequence_id;
                        if({rx_data,crc_low}!=rx_crc) event_status<=2;
                        else if(blocked || occupied) event_status<=5;
                        else begin event_status<=1; core_start<=1; image_blank<=!any_ink; end
                    end
                    default: rx_state<=SYNC0;
                endcase
            end
        end
    end

    // Byte-at-a-time response transmitter. Result data stays stable until this
    // packet completes; occupied prevents the image buffer from being reused.
    assign tx_start=tx_state==1;
    always @* begin
        score_address=0;
        if(tx_index>=19 && tx_index<=58) score_address=(tx_index-19)>>2;
        tx_data=0;
        case(tx_index)
            0: tx_data=8'ha5;
            1: tx_data=8'h5a;
            2: tx_data=1;
            3: tx_data=8'h81;
            4: tx_data=tx_sequence;
            5: tx_data=52;
            6: tx_data=0;
            7: tx_data=status;
            8: tx_data=status==0 ? {4'd0,core_digit} : 8'hff;
            9: tx_data=status==0 ? {7'd0,core_unknown} : 0;
            10: tx_data=0;
            11,12,13,14: tx_data=status==0 ? core_cycles >> ((tx_index-11)*8) : 0;
            15,16,17,18: tx_data=status==0 ? core_margin >> ((tx_index-15)*8) : 0;
            59: tx_data=tx_crc[7:0];
            60: tx_data=tx_crc[15:8];
            default: if(tx_index>=19 && tx_index<=58 && status==0)
                         tx_data=score_data >> (((tx_index-19)&3)*8);
        endcase
    end
    always @(posedge clk) begin
        if(reset) begin tx_state<=0; tx_index<=0; status<=0; tx_sequence<=0; tx_crc<=16'hffff; end
        else case(tx_state)
            0: if(pop_fifo || take_result) begin
                   status<=pop_fifo ? fifo_status[read_pointer] : 0;
                   tx_sequence<=pop_fifo ? fifo_sequence[read_pointer] : result_sequence;
                   tx_index<=0; tx_crc<=16'hffff; tx_state<=1;
               end
            1: begin
                   if(tx_index>=2 && tx_index<=58) tx_crc<=crc_byte(tx_crc,tx_data);
                   tx_state<=2;
               end
            2: if(tx_busy) tx_state<=3;
            3: if(!tx_busy) begin
                   if(tx_index==60) tx_state<=0;
                   else begin tx_index<=tx_index+1'b1; tx_state<=1; end
               end
            default: tx_state<=0;
        endcase
    end
endmodule
