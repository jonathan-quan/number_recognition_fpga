// Independent AI MAC; the original processing_element.v is never modified.
module ai_mac (
    input wire clk, reset, load_bias, enable,
    input wire signed [31:0] bias,
    input wire signed [7:0] a, b,
    output reg signed [31:0] accumulator
);
    (* use_dsp = "yes" *) wire signed [15:0] product;
    assign product = a * b;
    always @(posedge clk) begin
        if (reset) accumulator <= 32'sd0;
        else if (load_bias) accumulator <= bias;
        else if (enable) accumulator <= accumulator + {{16{product[15]}}, product};
    end
endmodule
