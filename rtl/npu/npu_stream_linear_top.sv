`timescale 1ns/1ps
`default_nettype none
// Parameters accompany each result in tagged tile order; no full-layer table.
module npu_stream_linear_top #(
    parameter integer LANES=4,K_TILE=16
) (
    input wire clk,rst,cmd_valid,
    output wire cmd_ready,
    input wire [15:0] cmd_m,cmd_n,cmd_k,
    input wire cmd_relu,
    input wire operand_valid,
    output wire operand_ready,
    input wire [LANES*8-1:0] a_vector,b_vector,
    input wire param_valid,
    output wire param_ready,
    input wire signed [31:0] param_bias,param_multiplier,
    input wire [5:0] param_shift,
    output wire busy,
    output reg done,
    output wire error,
    output wire out_valid,
    input wire out_ready,
    output wire [15:0] out_row,out_col,
    output wire signed [7:0] out_data,
    output wire out_last,
    output reg [63:0] stat_cycles,
    output wire [63:0] stat_load_cycles,stat_compute_cycles,stat_input_wait_cycles,
    output wire [63:0] stat_buffer_stall_cycles,
    output reg [63:0] stat_output_stall_cycles,
    output wire [63:0] stat_overlap_cycles,stat_mac_count
);
    reg active,relu_latched,param_error,core_finished,valid_slot;
    reg [15:0] row_slot,col_slot;
    reg signed [7:0] data_slot;
    reg last_slot;
    wire raw_cmd_ready,raw_done,raw_error,raw_valid,raw_ready,raw_last;
    wire [15:0] raw_row,raw_col;
    wire signed [31:0] raw_data;
    wire slot_ready=!valid_slot || out_ready;
    wire param_good=param_multiplier>0 && param_shift<=31;
    wire command_fire=cmd_valid && cmd_ready;
    wire signed [32:0] biased_sum={raw_data[31],raw_data}+{param_bias[31],param_bias};
    wire signed [63:0] product=biased_sum*param_multiplier;
    wire [63:0] magnitude=product[63]?(~product+64'd1):product;
    wire [63:0] rounding=param_shift==0?64'd0:(64'd1<<(param_shift-1'b1));
    wire [63:0] rounded=(magnitude+rounding)>>param_shift;
    wire signed [63:0] scaled=product[63]?-$signed(rounded):$signed(rounded);
    wire signed [63:0] activated=relu_latched && scaled<0?64'sd0:scaled;
    wire signed [7:0] quantized=activated>127?8'sd127:activated< -128?-8'sd128:activated[7:0];
    assign cmd_ready=!rst && !active && raw_cmd_ready;
    assign busy=active;
    assign error=raw_error || param_error;
    assign out_valid=!rst && valid_slot;
    assign out_row=row_slot;assign out_col=col_slot;assign out_data=data_slot;assign out_last=last_slot;
    assign param_ready=!rst && raw_valid && slot_ready;
    assign raw_ready=param_ready && param_valid && param_good;
    npu_stream_top #(.LANES(LANES),.K_TILE(K_TILE)) gemm (
        .clk(clk),.rst(rst),.cmd_valid(command_fire),.cmd_ready(raw_cmd_ready),
        .cmd_m(cmd_m),.cmd_n(cmd_n),.cmd_k(cmd_k),.operand_valid(operand_valid),
        .operand_ready(operand_ready),.a_vector(a_vector),.b_vector(b_vector),
        .busy(),.done(raw_done),.error(raw_error),.out_valid(raw_valid),.out_ready(raw_ready),
        .out_row(raw_row),.out_col(raw_col),.out_data(raw_data),.out_last(raw_last),
        .stat_cycles(),.stat_load_cycles(stat_load_cycles),.stat_compute_cycles(stat_compute_cycles),
        .stat_input_wait_cycles(stat_input_wait_cycles),.stat_buffer_stall_cycles(stat_buffer_stall_cycles),
        .stat_output_stall_cycles(),.stat_overlap_cycles(stat_overlap_cycles),.stat_mac_count(stat_mac_count)
    );
    always @(posedge clk) begin
        if(rst) begin
            active<=0;relu_latched<=0;param_error<=0;core_finished<=0;valid_slot<=0;
            row_slot<=0;col_slot<=0;data_slot<=0;last_slot<=0;done<=0;
            stat_cycles<=0;stat_output_stall_cycles<=0;
        end else begin
            done<=0;param_error<=0;
            if(command_fire && cmd_m!=0 && cmd_n!=0 && cmd_k!=0) begin
                active<=1;relu_latched<=cmd_relu;core_finished<=0;
                stat_cycles<=0;stat_output_stall_cycles<=0;
            end
            if(active) begin
                stat_cycles<=stat_cycles+1'b1;
                if(out_valid && !out_ready) stat_output_stall_cycles<=stat_output_stall_cycles+1'b1;
                if(raw_done) core_finished<=1;
                if((core_finished || raw_done) && (!valid_slot || out_ready)) begin
                    active<=0;core_finished<=0;done<=1;
                end
            end
            if(valid_slot && out_ready) valid_slot<=0;
            if(param_valid && param_ready && !param_good) param_error<=1;
            if(raw_valid && raw_ready) begin
                valid_slot<=1;row_slot<=raw_row;col_slot<=raw_col;data_slot<=quantized;last_slot<=raw_last;
            end
        end
    end
endmodule
`default_nettype wire
