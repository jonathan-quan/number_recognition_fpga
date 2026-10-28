// Exact 784 -> 128 -> 64 -> 10 integer MLP from export/manifest.json.
// Three cycles per MAC: synchronous ROM read, operand register, accumulation.
module ai_inference (
    input wire clk, reset, start,
    input wire pixel_write,
    input wire [9:0] pixel_address,
    input wire [7:0] pixel_data,
    input wire image_blank,
    output wire busy,
    output reg done,
    output reg [3:0] digit,
    output reg unknown,
    output reg [31:0] margin,
    output reg [31:0] inference_cycles,
    input wire [3:0] score_address,
    output wire signed [31:0] score_data,
    output wire trace_valid,
    output wire [1:0] trace_layer,
    output wire [7:0] trace_neuron,
    output wire signed [31:0] trace_value
);
    localparam IDLE=0, BIAS_WAIT=1, BIAS_LOAD=2, READ=3, MAC=4,
               STORE=5, ARG_INIT=6, ARG_SCAN=7, FINISH=8,
               CONF_INIT=9, CONF_READ=10, CONF_ADD=11, OPERANDS=12;
    reg [3:0] state;
    reg [1:0] layer;
    reg [7:0] neuron;
    reg [9:0] index;
    reg [16:0] base;
    reg blank;
    reg [31:0] cycle_count;
    wire [9:0] input_count = layer==0 ? 10'd784 : layer==1 ? 10'd128 : 10'd64;
    wire [7:0] output_count = layer==0 ? 8'd128 : layer==1 ? 8'd64 : 8'd10;
    wire [16:0] weight_address = base + index;

    // Small combinational normalization LUT; unlike weights, 256 bytes is
    // cheaper as distributed ROM than consuming a dedicated BRAM.
    (* rom_style="distributed" *) reg signed [7:0] input_lut [0:255];
    initial $readmemh("input_lut.mem", input_lut);
    (* ram_style="block" *) reg signed [7:0] image [0:783];
    reg signed [7:0] image_q;
    always @(posedge clk) begin
        if (pixel_write && !busy) image[pixel_address] <= input_lut[pixel_data];
        image_q <= image[index];
    end
    (* ram_style="distributed" *) reg signed [7:0] hidden1 [0:127];
    (* ram_style="distributed" *) reg signed [7:0] hidden2 [0:63];
    reg signed [31:0] scores [0:9];
    assign score_data = scores[score_address];

    wire signed [7:0] w1,w2,w3;
    wire signed [31:0] b1,b2,b3;
    ai_rom #(.DEPTH(100352),.ADDR_WIDTH(17),.FILE("fc1_weights.mem")) rw1(clk,weight_address,w1);
    ai_rom #(.DEPTH(8192),.ADDR_WIDTH(13),.FILE("fc2_weights.mem")) rw2(clk,weight_address[12:0],w2);
    ai_rom #(.DEPTH(640),.ADDR_WIDTH(10),.FILE("fc3_weights.mem")) rw3(clk,weight_address[9:0],w3);
    ai_rom #(.WIDTH(32),.DEPTH(128),.ADDR_WIDTH(7),.FILE("fc1_biases.mem"),.STYLE("distributed")) rb1(clk,neuron[6:0],b1);
    ai_rom #(.WIDTH(32),.DEPTH(64),.ADDR_WIDTH(6),.FILE("fc2_biases.mem"),.STYLE("distributed")) rb2(clk,neuron[5:0],b2);
    ai_rom #(.WIDTH(32),.DEPTH(10),.ADDR_WIDTH(4),.FILE("fc3_biases.mem"),.STYLE("distributed")) rb3(clk,neuron[3:0],b3);
    wire signed [7:0] weight = layer==0 ? w1 : layer==1 ? w2 : w3;
    wire signed [31:0] bias = layer==0 ? b1 : layer==1 ? b2 : b3;
    wire signed [7:0] activation = layer==0 ? image_q : layer==1 ? hidden1[index[6:0]] : hidden2[index[5:0]];
    reg signed [7:0] registered_activation,registered_weight;
    always @(posedge clk) begin
        if(reset) begin registered_activation<=0; registered_weight<=0; end
        else if(state==OPERANDS) begin registered_activation<=activation; registered_weight<=weight; end
    end
    wire signed [31:0] accumulator;
    ai_mac mac_unit(clk,reset,state==BIAS_LOAD,state==MAC,bias,registered_activation,registered_weight,accumulator);

    // ReLU before rounding: exact half-away-from-zero is an unsigned rounded
    // shift here. Use 33 bits for the rounding addition, then saturate.
    wire [32:0] positive_acc = accumulator[31] ? 33'd0 : {1'b0,accumulator};
    wire [32:0] rounded = layer==0 ? (positive_acc+33'd512)>>10 : (positive_acc+33'd128)>>8;
    wire [7:0] requantized = rounded>127 ? 8'd127 : rounded[7:0];
    assign trace_valid = state==STORE;
    assign trace_layer = layer;
    assign trace_neuron = neuron;
    assign trace_value = layer==2 ? accumulator : $signed({24'd0,requantized});
    assign busy = state!=IDLE;

    reg signed [31:0] best, second;
    reg [3:0] best_digit, scan;
    wire signed [32:0] score_gap = {best[31],best} - {second[31],second};
    reg [3:0] confidence_index;
    reg [19:0] exponential_sum;
    wire signed [31:0] confidence_score=scores[confidence_index];
    wire signed [32:0] confidence_delta={best[31],best}-{confidence_score[31],confidence_score};
    wire [16:0] exponential_q;
    // ceil(65536*exp(-floor(delta/16)/64)); Q10 input logits. Larger
    // deltas use 1, an upper bound beyond 16 real logit units. The sum is
    // therefore conservative: accepted inputs have softmax confidence >=90%.
    ai_rom #(.WIDTH(17),.DEPTH(1024),.ADDR_WIDTH(10),.FILE("confidence_exp.mem"))
        confidence_rom(clk,confidence_delta[13:4],exponential_q);
    wire [19:0] exponential_term=confidence_delta>=33'sd16384 ? 20'd1 : {3'd0,exponential_q};
    always @(posedge clk) begin
        if (reset) begin
            state<=IDLE; done<=0; layer<=0; neuron<=0; index<=0; base<=0;
            digit<=0; unknown<=1; margin<=0; cycle_count<=0; inference_cycles<=0;
            blank<=1; best<=0; second<=0; best_digit<=0; scan<=0;
            confidence_index<=0; exponential_sum<=0;
        end else begin
            done<=0;
            if (busy) cycle_count<=cycle_count+1'b1;
            case (state)
                IDLE: if (start) begin
                    layer<=0; neuron<=0; index<=0; base<=0; blank<=image_blank;
                    cycle_count<=0; state<=BIAS_WAIT;
                end
                BIAS_WAIT: state<=BIAS_LOAD;
                BIAS_LOAD: state<=READ;
                READ: state<=OPERANDS;
                OPERANDS: state<=MAC;
                MAC: if (index==input_count-1'b1) state<=STORE;
                     else begin index<=index+1'b1; state<=READ; end
                STORE: begin
                    if (layer==0) hidden1[neuron[6:0]]<=requantized;
                    else if (layer==1) hidden2[neuron[5:0]]<=requantized;
                    else scores[neuron[3:0]]<=accumulator;
                    index<=0;
                    if (neuron==output_count-1'b1) begin
                        neuron<=0; base<=0;
                        if (layer==2) state<=ARG_INIT;
                        else begin layer<=layer+1'b1; state<=BIAS_WAIT; end
                    end else begin neuron<=neuron+1'b1; base<=base+input_count; state<=BIAS_WAIT; end
                end
                ARG_INIT: begin
                    best<=scores[0]; second<=32'sh80000000; best_digit<=0;
                    scan<=1; state<=ARG_SCAN;
                end
                ARG_SCAN: begin
                    // Strict > preserves the lowest-index winner in a tie.
                    if (scores[scan]>best) begin second<=best; best<=scores[scan]; best_digit<=scan; end
                    else if (scores[scan]>second) second<=scores[scan];
                    if (scan==9) state<=CONF_INIT; else scan<=scan+1'b1;
                end
                CONF_INIT: begin confidence_index<=0; exponential_sum<=0; state<=CONF_READ; end
                CONF_READ: state<=CONF_ADD;
                CONF_ADD: begin
                    exponential_sum<=exponential_sum+exponential_term;
                    if(confidence_index==9) state<=FINISH;
                    else begin confidence_index<=confidence_index+1'b1; state<=CONF_READ; end
                end
                FINISH: begin
                    digit<=best_digit; unknown<=blank || exponential_sum>20'd72817;
                    margin<=score_gap[31:0]; inference_cycles<=cycle_count+1'b1;
                    done<=1; state<=IDLE;
                end
                default: state<=IDLE;
            endcase
        end
    end
endmodule
