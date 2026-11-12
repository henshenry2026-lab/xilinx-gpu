`timescale 1ns/1ps
`default_nettype none
// GEMM with a bounded tile working set. The caller replays vectors in
// tile-row, tile-column, K order; no complete matrix is cached here.
module npu_stream_top #(
    parameter integer LANES=4, K_TILE=16
) (
    input wire clk,rst,cmd_valid,
    output wire cmd_ready,
    input wire [15:0] cmd_m,cmd_n,cmd_k,
    input wire operand_valid,
    output wire operand_ready,
    input wire [LANES*8-1:0] a_vector,b_vector,
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
    reg active,load_complete,chunk_active;
    reg [15:0] m,n,depth,load_row,load_col,load_k,load_chunk;
    reg [15:0] tile_row,tile_col,k_offset;
    wire [16:0] next_load_row={1'b0,load_row}+LANES;
    wire [16:0] next_load_col={1'b0,load_col}+LANES;
    wire [16:0] next_tile_row={1'b0,tile_row}+LANES;
    wire [16:0] next_tile_col={1'b0,tile_col}+LANES;
    wire [15:0] tile_rows=m-tile_row<LANES?m-tile_row:LANES;
    wire [15:0] tile_cols=n-tile_col<LANES?n-tile_col:LANES;
    wire buf_in_ready,block_valid,block_ready,vec_valid,vec_ready;
    wire [15:0] block_count;
    wire [LANES*8-1:0] chunk_a,chunk_b;
    wire core_cmd_ready,core_done,core_error,compute_active,phase_active;
    wire core_cmd_valid=active && !chunk_active && block_valid;
    wire final_chunk={1'b0,k_offset}+{1'b0,block_count}=={1'b0,depth};
    wire final_tile=next_tile_row>={1'b0,m} && next_tile_col>={1'b0,n};
    wire load_fire=operand_valid && operand_ready;
    wire load_last=load_chunk==K_TILE-1 || load_k==depth-1'b1;
    reg [15:0] macs_per_vector;
    integer r,c;
    always @* begin
        macs_per_vector=0;
        for(r=0;r<LANES;r=r+1) for(c=0;c<LANES;c=c+1)
            if(r<tile_rows && c<tile_cols) macs_per_vector=macs_per_vector+1'b1;
    end
    assign cmd_ready=!rst && !active;
    assign busy=active;
    assign operand_ready=!rst && active && !load_complete && buf_in_ready;
    assign block_ready=core_cmd_valid && core_cmd_ready;
    npu_stream_buffer #(.LANES(LANES),.K_TILE(K_TILE)) buffer (
        .clk(clk),.rst(rst),.in_valid(operand_valid && active && !load_complete),
        .in_ready(buf_in_ready),.in_a(a_vector),.in_b(b_vector),.in_last(load_last),
        .block_valid(block_valid),.block_ready(block_ready),.block_count(block_count),
        .operand_valid(vec_valid),.operand_ready(vec_ready),.a_vector(chunk_a),.b_vector(chunk_b)
    );
    npu_tile_core #(.LANES(LANES),.K_TILE(K_TILE)) core (
        .clk(clk),.rst(rst),.cmd_valid(core_cmd_valid),.cmd_ready(core_cmd_ready),
        .base_row(tile_row),.base_col(tile_col),.rows_valid(tile_rows),.cols_valid(tile_cols),
        .k_count(block_count),.clear_accum(k_offset==0),.emit_result(final_chunk),
        .job_last(final_chunk && final_tile),.operand_valid(vec_valid),.operand_ready(vec_ready),
        .a_vector(chunk_a),.b_vector(chunk_b),.a_mask({LANES{1'b1}}),.b_mask({LANES{1'b1}}),
        .busy(),.done(core_done),.error(core_error),.out_valid(out_valid),.out_ready(out_ready),
        .out_row(out_row),.out_col(out_col),.out_data(out_data),.out_last(out_last),
        .compute_active(compute_active),.phase_active(phase_active)
    );
    // block_count changes when its reader bank is released; retain descriptor state.
    reg [15:0] running_count;
    reg running_final_chunk,running_final_tile;
    always @(posedge clk) begin
        if(rst) begin
            active<=0;load_complete<=0;chunk_active<=0;done<=0;error<=0;
            m<=0;n<=0;depth<=0;load_row<=0;load_col<=0;load_k<=0;load_chunk<=0;
            tile_row<=0;tile_col<=0;k_offset<=0;running_count<=0;
            running_final_chunk<=0;running_final_tile<=0;
            stat_cycles<=0;stat_load_cycles<=0;stat_compute_cycles<=0;
            stat_input_wait_cycles<=0;stat_buffer_stall_cycles<=0;
            stat_output_stall_cycles<=0;stat_overlap_cycles<=0;stat_mac_count<=0;
        end else begin
            done<=0;error<=core_error;
            if(cmd_valid && cmd_ready) begin
                if(cmd_m==0 || cmd_n==0 || cmd_k==0) error<=1;
                else begin
                    active<=1;m<=cmd_m;n<=cmd_n;depth<=cmd_k;
                    load_complete<=0;chunk_active<=0;
                    load_row<=0;load_col<=0;load_k<=0;load_chunk<=0;
                    tile_row<=0;tile_col<=0;k_offset<=0;
                    stat_cycles<=0;stat_load_cycles<=0;stat_compute_cycles<=0;
                    stat_input_wait_cycles<=0;stat_buffer_stall_cycles<=0;
                    stat_output_stall_cycles<=0;stat_overlap_cycles<=0;stat_mac_count<=0;
                end
            end
            if(active) begin
                stat_cycles<=stat_cycles+1'b1;
                if(load_fire) stat_load_cycles<=stat_load_cycles+1'b1;
                if(phase_active) stat_compute_cycles<=stat_compute_cycles+1'b1;
                if(operand_ready && !operand_valid) stat_input_wait_cycles<=stat_input_wait_cycles+1'b1;
                if(operand_valid && !load_complete && !buf_in_ready)
                    stat_buffer_stall_cycles<=stat_buffer_stall_cycles+1'b1;
                if(out_valid && !out_ready) stat_output_stall_cycles<=stat_output_stall_cycles+1'b1;
                if(load_fire && phase_active) stat_overlap_cycles<=stat_overlap_cycles+1'b1;
                if(vec_valid && vec_ready) stat_mac_count<=stat_mac_count+macs_per_vector;
                if(load_fire) begin
                    load_chunk<=load_last?0:load_chunk+1'b1;
                    if(load_k==depth-1'b1) begin
                        load_k<=0;
                        if(next_load_col<{1'b0,n}) load_col<=next_load_col[15:0];
                        else begin
                            load_col<=0;
                            if(next_load_row<{1'b0,m}) load_row<=next_load_row[15:0];
                            else load_complete<=1;
                        end
                    end else load_k<=load_k+1'b1;
                end
                if(block_ready) begin
                    chunk_active<=1;running_count<=block_count;
                    running_final_chunk<=final_chunk;running_final_tile<=final_tile;
                end
                if(core_done) begin
                    chunk_active<=0;
                    if(running_final_chunk) begin
                        k_offset<=0;
                        if(running_final_tile) begin active<=0;done<=1;end
                        else if(next_tile_col<{1'b0,n}) tile_col<=next_tile_col[15:0];
                        else begin tile_col<=0;tile_row<=next_tile_row[15:0];end
                    end else k_offset<=k_offset+running_count;
                end
            end
        end
    end
endmodule
`default_nettype wire
