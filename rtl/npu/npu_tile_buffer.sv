`timescale 1ns/1ps
`default_nettype none

// Bank A by row and B by column. Each bank has one synchronous read port.
// Storage is intentionally not reset; all operands used by a job must be loaded.
module npu_tile_buffer #(
    parameter integer LANES = 4,
    parameter integer MAX_M = 16,
    parameter integer MAX_N = 16,
    parameter integer MAX_K = 64
) (
    input wire clk,
    input wire wr_en, wr_sel,
    input wire [15:0] wr_addr,
    input wire signed [7:0] wr_data,
    input wire rd_en,
    input wire [15:0] a_rd_addr, b_rd_addr,
    input wire [LANES-1:0] a_mask, b_mask,
    output reg [LANES*8-1:0] a_data, b_data
);
    localparam integer ROW_GROUPS = (MAX_M + LANES - 1) / LANES;
    localparam integer COL_GROUPS = (MAX_N + LANES - 1) / LANES;
    localparam integer A_DEPTH = ROW_GROUPS * MAX_K;
    localparam integer B_DEPTH = COL_GROUPS * MAX_K;
    wire [15:0] a_write_bank = (wr_addr / MAX_K) % LANES;
    wire [15:0] a_write_addr = (wr_addr / (MAX_K * LANES)) * MAX_K + wr_addr % MAX_K;
    wire [15:0] b_write_bank = (wr_addr % MAX_N) % LANES;
    wire [15:0] b_write_addr = (wr_addr / MAX_N) * COL_GROUPS + (wr_addr % MAX_N) / LANES;
    genvar bank;
    generate
        for (bank = 0; bank < LANES; bank = bank + 1) begin: banks
            (* ram_style = "block" *) reg signed [7:0] a_mem [0:A_DEPTH-1];
            (* ram_style = "block" *) reg signed [7:0] b_mem [0:B_DEPTH-1];
            always @(posedge clk) begin
                if (wr_en && !wr_sel && a_write_bank == bank) a_mem[a_write_addr] <= wr_data;
                if (wr_en &&  wr_sel && b_write_bank == bank) b_mem[b_write_addr] <= wr_data;
                if (rd_en) begin
                    a_data[bank*8 +: 8] <= a_mask[bank] ? a_mem[a_rd_addr] : 8'd0;
                    b_data[bank*8 +: 8] <= b_mask[bank] ? b_mem[b_rd_addr] : 8'd0;
                end
            end
        end
    endgenerate
endmodule
`default_nettype wire
