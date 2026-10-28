// BCD adaptation of the original display's active-low abcdefg truth table.
// Avoid binary-to-decimal division: only 000d or 9999 is ever displayed.
module ai_display (
    input wire clk, reset,
    input wire [15:0] bcd,
    output reg [6:0] segments,
    output reg [3:0] an,
    output wire dp
);
    reg [16:0] refresh;
    reg [3:0] digit;
    assign dp=1'b1;
    always @(posedge clk) if(reset) refresh<=0; else refresh<=refresh+1'b1;
    always @* begin
        case(refresh[16:15])
            0: begin an=4'b1110; digit=bcd[3:0]; end
            1: begin an=4'b1101; digit=bcd[7:4]; end
            2: begin an=4'b1011; digit=bcd[11:8]; end
            default: begin an=4'b0111; digit=bcd[15:12]; end
        endcase
        case(digit)
            0: segments=7'b0000001;
            1: segments=7'b1001111;
            2: segments=7'b0010010;
            3: segments=7'b0000110;
            4: segments=7'b1001100;
            5: segments=7'b0100100;
            6: segments=7'b0100000;
            7: segments=7'b0001111;
            8: segments=7'b0000000;
            9: segments=7'b0000100;
            default: segments=7'b1111111;
        endcase
    end
endmodule
