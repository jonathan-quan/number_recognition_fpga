// Synchronous, one-cycle-read ROM. Large weight instances infer block RAM.
module ai_rom #(
    parameter WIDTH = 8,
    parameter DEPTH = 256,
    parameter ADDR_WIDTH = 8,
    parameter FILE = "input_lut.mem",
    parameter STYLE = "block"
) (
    input wire clk,
    input wire [ADDR_WIDTH-1:0] addr,
    output reg signed [WIDTH-1:0] data
);
    (* rom_style = STYLE *) reg signed [WIDTH-1:0] memory [0:DEPTH-1];
    initial $readmemh(FILE, memory);
    always @(posedge clk) data <= memory[addr];
endmodule
