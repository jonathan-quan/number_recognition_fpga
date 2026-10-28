// 8-N-1, integer divider. Default: 100 MHz / 115200 -> 868 clocks/bit.
module ai_uart_rx #(parameter CLOCK_HZ=100000000, parameter BAUD=115200) (
    input wire clk, reset, rx,
    output reg [7:0] data,
    output reg valid, framing_error
);
    localparam CPB=CLOCK_HZ/BAUD;
    (* ASYNC_REG="TRUE" *) reg [1:0] rx_sync=2'b11;
    reg [1:0] state;
    reg [31:0] ticks;
    reg [2:0] bit_index;
    reg [7:0] shift;
    always @(posedge clk) begin
        rx_sync<={rx_sync[0],rx};
        if(reset) begin state<=0; ticks<=0; bit_index<=0; shift<=0; data<=0; valid<=0; framing_error<=0; rx_sync<=3; end
        else begin
            valid<=0; framing_error<=0;
            case(state)
                0: if(!rx_sync[1]) begin ticks<=CPB/2-1; state<=1; end
                1: if(ticks!=0) ticks<=ticks-1'b1;
                   else if(!rx_sync[1]) begin ticks<=CPB-1; bit_index<=0; state<=2; end
                   else state<=0;
                2: if(ticks!=0) ticks<=ticks-1'b1;
                   else begin
                       shift[bit_index]<=rx_sync[1]; ticks<=CPB-1;
                       if(bit_index==7) state<=3; else bit_index<=bit_index+1'b1;
                   end
                3: if(ticks!=0) ticks<=ticks-1'b1;
                   else begin
                       if(rx_sync[1]) begin data<=shift; valid<=1; end
                       else framing_error<=1;
                       state<=0;
                   end
            endcase
        end
    end
endmodule

module ai_uart_tx #(parameter CLOCK_HZ=100000000, parameter BAUD=115200) (
    input wire clk, reset, start,
    input wire [7:0] data,
    output wire tx,
    output reg busy
);
    localparam CPB=CLOCK_HZ/BAUD;
    reg [31:0] ticks;
    reg [9:0] frame;
    reg [3:0] bit_index;
    assign tx=busy ? frame[bit_index] : 1'b1;
    always @(posedge clk) begin
        if(reset) begin busy<=0; ticks<=0; frame<=10'h3ff; bit_index<=0; end
        else if(!busy) begin
            if(start) begin frame<={1'b1,data,1'b0}; bit_index<=0; ticks<=CPB-1; busy<=1; end
        end else if(ticks!=0) ticks<=ticks-1'b1;
        else begin
            ticks<=CPB-1;
            if(bit_index==9) busy<=0; else bit_index<=bit_index+1'b1;
        end
    end
endmodule
