`timescale 1ns/1ps
`default_nettype none
// One vector scalar is broadcast to LANES output accumulators per accepted beat.
// The caller streams output groups in ascending column order, then ascending K.
// Only the current output group is retained; no input/weight RAM is required.
module npu_gemv_top #(
    parameter integer LANES=4
) (
    input wire clk,rst,cmd_valid,
    output wire cmd_ready,
    input wire [15:0] cmd_n,cmd_k,
    input wire operand_valid,
    output wire operand_ready,
    input wire signed [7:0] a_scalar,
    input wire [LANES*8-1:0] b_vector,
    output wire busy,
    output reg done,error,
    output wire out_valid,
    input wire out_ready,
    output wire [15:0] out_row,out_col,
    output wire signed [31:0] out_data,
    output wire out_last,
    output reg [63:0] stat_cycles,stat_load_cycles,stat_compute_cycles,
    output reg [63:0] stat_input_wait_cycles,stat_buffer_stall_cycles,
    output reg [63:0] stat_output_stall_cycles,stat_overlap_cycles,stat_mac_count
);
    localparam integer LANE_BITS=LANES>1?$clog2(LANES):1;
    reg active,emitting;
    reg [15:0] n,depth,base_col,k_index;
    reg [LANE_BITS-1:0] emit_lane;
    reg signed [31:0] accum[0:LANES-1];
    wire [16:0] next_group={1'b0,base_col}+LANES;
    wire [16:0] current_col={1'b0,base_col}+emit_lane;
    wire [15:0] active_lanes=n-base_col<LANES?n-base_col:LANES;
    wire operand_fire=operand_valid && operand_ready;
    wire output_fire=out_valid && out_ready;
    wire group_last=emit_lane==LANES-1 || current_col+17'd1>={1'b0,n};
    wire final_group=next_group>={1'b0,n};

    assign cmd_ready=!rst && !active;
    assign busy=active;
    assign operand_ready=!rst && active && !emitting;
    assign out_valid=!rst && active && emitting;
    assign out_row=16'd0;
    assign out_col=current_col[15:0];
    assign out_data=accum[emit_lane];
    assign out_last=out_valid && current_col+17'd1=={1'b0,n};

    genvar lane;
    generate for(lane=0;lane<LANES;lane=lane+1) begin: mac_lane
        wire signed [7:0] weight=b_vector[lane*8+:8];
        wire signed [15:0] product=a_scalar*weight;
        // Keeping the multiplier at 8x8 gives one signed DSP MAC per lane.
        always @(posedge clk) begin
            if(rst) accum[lane]<=0;
            else if(cmd_valid && cmd_ready && cmd_n!=0 && cmd_k!=0)
                accum[lane]<=0;
            else if(output_fire && group_last) accum[lane]<=0;
            else if(operand_fire && lane<active_lanes)
                accum[lane]<=accum[lane]+{{16{product[15]}},product};
        end
    end endgenerate

    always @(posedge clk) begin
        if(rst) begin
            active<=0;emitting<=0;n<=0;depth<=0;base_col<=0;k_index<=0;emit_lane<=0;
            done<=0;error<=0;
            stat_cycles<=0;stat_load_cycles<=0;stat_compute_cycles<=0;
            stat_input_wait_cycles<=0;stat_buffer_stall_cycles<=0;
            stat_output_stall_cycles<=0;stat_overlap_cycles<=0;stat_mac_count<=0;
        end else begin
            done<=0;error<=0;
            if(cmd_valid && cmd_ready) begin
                if(cmd_n==0 || cmd_k==0) error<=1;
                else begin
                    active<=1;emitting<=0;n<=cmd_n;depth<=cmd_k;
                    base_col<=0;k_index<=0;emit_lane<=0;
                    stat_cycles<=0;stat_load_cycles<=0;stat_compute_cycles<=0;
                    stat_input_wait_cycles<=0;stat_buffer_stall_cycles<=0;
                    stat_output_stall_cycles<=0;stat_overlap_cycles<=0;stat_mac_count<=0;
                end
            end
            if(active) begin
                stat_cycles<=stat_cycles+1'b1;
                if(operand_ready && !operand_valid)
                    stat_input_wait_cycles<=stat_input_wait_cycles+1'b1;
                if(out_valid && !out_ready)
                    stat_output_stall_cycles<=stat_output_stall_cycles+1'b1;
                if(operand_fire) begin
                    stat_load_cycles<=stat_load_cycles+1'b1;
                    stat_compute_cycles<=stat_compute_cycles+1'b1;
                    stat_overlap_cycles<=stat_overlap_cycles+1'b1;
                    stat_mac_count<=stat_mac_count+active_lanes;
                    if(k_index==depth-1'b1) begin
                        k_index<=0;emit_lane<=0;emitting<=1;
                    end else k_index<=k_index+1'b1;
                end
                if(output_fire) begin
                    if(group_last) begin
                        emit_lane<=0;
                        if(final_group) begin active<=0;emitting<=0;done<=1;end
                        else begin base_col<=next_group[15:0];emitting<=0;end
                    end else emit_lane<=emit_lane+1'b1;
                end
            end
        end
    end
endmodule
`default_nettype wire
