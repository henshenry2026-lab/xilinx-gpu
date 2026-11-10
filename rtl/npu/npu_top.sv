`timescale 1ns/1ps
`default_nettype none

// C[M,N] = A[M,K] * B[K,N], signed INT8 inputs and exact INT32 outputs.
// Writes use fixed strides MAX_K (A) and MAX_N (B), independent of job shape.
module npu_top #(
    parameter integer LANES = 4,
    parameter integer MAX_M = 16,
    parameter integer MAX_N = 16,
    parameter integer MAX_K = 64
) (
    input wire clk, rst,
    input wire wr_valid,
    output wire wr_ready,
    input wire wr_sel,
    input wire [15:0] wr_addr,
    input wire signed [7:0] wr_data,
    input wire cmd_valid,
    output wire cmd_ready,
    input wire [15:0] cmd_m, cmd_n, cmd_k,
    output wire busy,
    output reg done, error,
    output wire out_valid,
    input wire out_ready,
    output wire [15:0] out_row, out_col,
    output wire signed [31:0] out_data,
    output wire out_last
);
    localparam [2:0] IDLE = 0, CLEAR = 1, FEED = 2, DRAIN = 3, EMIT = 4;
    localparam integer LANE_BITS = LANES > 1 ? $clog2(LANES) : 1;
    localparam integer COL_GROUPS = (MAX_N + LANES - 1) / LANES;
    reg [2:0] state;
    reg [15:0] rows, cols, inner, tile_row, tile_col, feed_index, drain_count;
    reg [LANE_BITS-1:0] emit_row, emit_col;
    reg [LANES-1:0] a_mask, b_mask, read_a_mask, read_b_mask;
    reg read_valid;
    wire rd_en = state == FEED;
    wire [15:0] a_rd_addr = (tile_row / LANES) * MAX_K + feed_index;
    wire [15:0] b_rd_addr = feed_index * COL_GROUPS + tile_col / LANES;
    wire [LANES*8-1:0] a_data, b_data;
    wire [LANES*LANES*32-1:0] sums;
    wire write_in_range = wr_sel ? wr_addr < MAX_K*MAX_N : wr_addr < MAX_M*MAX_K;
    wire descriptor_valid = cmd_m != 0 && cmd_n != 0 && cmd_k != 0
        && cmd_m <= MAX_M && cmd_n <= MAX_N && cmd_k <= MAX_K;
    wire emit_in_bounds = out_row < rows && out_col < cols;
    integer lane;

    assign cmd_ready = !rst && state == IDLE;
    // Descriptors take priority over writes presented on the same clock.
    assign wr_ready = cmd_ready && !cmd_valid;
    assign busy = state != IDLE;
    assign out_valid = !rst && state == EMIT && emit_in_bounds;
    assign out_row = tile_row + emit_row;
    assign out_col = tile_col + emit_col;
    assign out_data = sums[(emit_row*LANES+emit_col)*32 +: 32];
    assign out_last = out_valid && out_row == rows-1'b1 && out_col == cols-1'b1;

    always @* begin
        a_mask = 0; b_mask = 0;
        for (lane = 0; lane < LANES; lane = lane + 1) begin
            a_mask[lane] = tile_row + lane < rows;
            b_mask[lane] = tile_col + lane < cols;
        end
    end
    // Align masks/valid with the synchronous BRAM outputs.
    always @(posedge clk) begin
        if (rst) begin
            read_valid <= 0; read_a_mask <= 0; read_b_mask <= 0;
        end else begin
            read_valid <= rd_en;
            if (rd_en) begin read_a_mask <= a_mask; read_b_mask <= b_mask; end
        end
    end

    npu_tile_buffer #(.LANES(LANES), .MAX_M(MAX_M), .MAX_N(MAX_N), .MAX_K(MAX_K)) buffer (
        .clk(clk), .wr_en(wr_valid && wr_ready && write_in_range), .wr_sel(wr_sel),
        .wr_addr(wr_addr), .wr_data(wr_data), .rd_en(rd_en), .a_rd_addr(a_rd_addr),
        .b_rd_addr(b_rd_addr), .a_mask(a_mask), .b_mask(b_mask), .a_data(a_data), .b_data(b_data)
    );
    npu_systolic_array #(.LANES(LANES)) array (
        .clk(clk), .rst(rst), .clear(state == CLEAR), .step(state == FEED || state == DRAIN),
        .a_data(a_data), .b_data(b_data), .a_valid(read_a_mask & {LANES{read_valid}}),
        .b_valid(read_b_mask & {LANES{read_valid}}), .sums(sums)
    );

    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE; done <= 0; error <= 0;
            rows <= 0; cols <= 0; inner <= 0; tile_row <= 0; tile_col <= 0;
            feed_index <= 0; drain_count <= 0; emit_row <= 0; emit_col <= 0;
        end else begin
            done <= 0; error <= 0;
            case (state)
                IDLE: if (cmd_valid) begin
                    if (descriptor_valid) begin
                        rows <= cmd_m; cols <= cmd_n; inner <= cmd_k;
                        tile_row <= 0; tile_col <= 0; state <= CLEAR;
                    end else error <= 1;
                end else if (wr_valid && wr_ready && !write_in_range) error <= 1;
                CLEAR: begin
                    feed_index <= 0; drain_count <= 0;
                    emit_row <= 0; emit_col <= 0; state <= FEED;
                end
                FEED: if (feed_index == inner-1'b1) state <= DRAIN;
                else feed_index <= feed_index + 1'b1;
                // First drain clock consumes the last BRAM read; allow the wavefront
                // another 2*(LANES-1) hops to reach the bottom-right accumulator.
                DRAIN: if (drain_count == 2*(LANES-1)) state <= EMIT;
                else drain_count <= drain_count + 1'b1;
                EMIT: if (!emit_in_bounds || out_ready) begin
                    if (emit_col != LANES-1) emit_col <= emit_col + 1'b1;
                    else begin
                        emit_col <= 0;
                        if (emit_row != LANES-1) emit_row <= emit_row + 1'b1;
                        else begin
                            emit_row <= 0;
                            if (tile_col + LANES < cols) begin
                                tile_col <= tile_col + LANES; state <= CLEAR;
                            end else if (tile_row + LANES < rows) begin
                                tile_col <= 0; tile_row <= tile_row + LANES; state <= CLEAR;
                            end else begin state <= IDLE; done <= 1; end
                        end
                    end
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
`default_nettype wire
