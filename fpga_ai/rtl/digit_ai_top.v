module digit_ai_top #(parameter CLOCK_HZ=100000000, parameter BAUD=115200,
                      parameter PACKET_TIMEOUT_CYCLES=10000000) (
    input wire clk, reset, uart_rx,
    output wire uart_tx,
    output wire [6:0] segments,
    output wire [3:0] an,
    output wire dp
);
    (* ASYNC_REG="TRUE" *) reg [1:0] reset_sync=2'b11;
    always @(posedge clk) reset_sync<={reset_sync[0],reset};
    wire rst=reset_sync[1];
    wire [7:0] rx_data,tx_data,pixel_data;
    wire rx_valid,rx_error,tx_start,tx_busy,start,busy,done,pixel_write,image_blank,unknown;
    wire [9:0] pixel_address;
    wire [3:0] digit,score_address;
    wire [31:0] cycles,margin;
    wire signed [31:0] score_data;
    ai_uart_rx #(.CLOCK_HZ(CLOCK_HZ),.BAUD(BAUD)) rx_unit(clk,rst,uart_rx,rx_data,rx_valid,rx_error);
    ai_uart_tx #(.CLOCK_HZ(CLOCK_HZ),.BAUD(BAUD)) tx_unit(clk,rst,tx_start,tx_data,uart_tx,tx_busy);
    ai_protocol #(.TIMEOUT_CYCLES(PACKET_TIMEOUT_CYCLES)) protocol_unit(
        .clk(clk),.reset(rst),.rx_valid(rx_valid),.rx_error(rx_error),.rx_data(rx_data),
        .tx_start(tx_start),.tx_data(tx_data),.tx_busy(tx_busy),
        .core_busy(busy),.core_done(done),.core_digit(digit),.core_unknown(unknown),
        .core_cycles(cycles),.core_margin(margin),.score_address(score_address),.score_data(score_data),
        .core_start(start),.pixel_write(pixel_write),.pixel_address(pixel_address),.pixel_data(pixel_data),.image_blank(image_blank));
    ai_inference core(
        .clk(clk),.reset(rst),.start(start),.pixel_write(pixel_write),.pixel_address(pixel_address),
        .pixel_data(pixel_data),.image_blank(image_blank),.busy(busy),.done(done),.digit(digit),
        .unknown(unknown),.margin(margin),.inference_cycles(cycles),.score_address(score_address),
        .score_data(score_data),.trace_valid(),.trace_layer(),.trace_neuron(),.trace_value());
    reg [15:0] display_bcd;
    always @(posedge clk) begin
        if(rst) display_bcd<=16'h9999;
        else if(done) display_bcd<=unknown ? 16'h9999 : {12'd0,digit};
    end
    ai_display display_unit(clk,rst,display_bcd,segments,an,dp);
endmodule
