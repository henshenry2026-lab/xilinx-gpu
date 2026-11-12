`timescale 1ns/1ps
`default_nettype none
// Two ownership-tracked chunk banks. Memory is deliberately not reset.
// A bank is reusable once its final vector is consumed, not when compute finishes.
module npu_stream_buffer #(
    parameter integer LANES=4, K_TILE=16
) (
    input wire clk,rst,
    input wire in_valid,
    output wire in_ready,
    input wire [LANES*8-1:0] in_a,in_b,
    input wire in_last,
    output wire block_valid,
    input wire block_ready,
    output wire [15:0] block_count,
    output wire operand_valid,
    input wire operand_ready,
    output wire [LANES*8-1:0] a_vector,b_vector
);
    localparam IW=K_TILE<=1?1:$clog2(K_TILE);
    (* ram_style="distributed" *) reg [LANES*8-1:0] a_mem[0:2*K_TILE-1];
    (* ram_style="distributed" *) reg [LANES*8-1:0] b_mem[0:2*K_TILE-1];
    reg [1:0] full;
    reg write_bank,read_bank,reading;
    reg [IW-1:0] wi,ri;
    reg [15:0] counts[0:1];
    wire [IW:0] write_address=(write_bank?K_TILE:0)+wi;
    wire [IW:0] read_address=(read_bank?K_TILE:0)+ri;
    assign in_ready=!rst && !full[write_bank];
    assign block_valid=!rst && full[read_bank] && !reading;
    assign block_count=counts[read_bank];
    assign operand_valid=!rst && reading;
    assign a_vector=a_mem[read_address];
    assign b_vector=b_mem[read_address];
    always @(posedge clk) begin
        if(rst) begin
            full<=0;write_bank<=0;read_bank<=0;reading<=0;wi<=0;ri<=0;
            counts[0]<=0;counts[1]<=0;
        end else begin
            if(in_valid && in_ready) begin
                a_mem[write_address]<=in_a;b_mem[write_address]<=in_b;
                if(in_last) begin
                    full[write_bank]<=1;
                    counts[write_bank]<={1'b0,wi}+16'd1;
                    write_bank<=!write_bank;wi<=0;
                end else wi<=wi+1'b1;
            end
            if(block_valid && block_ready) begin reading<=1;ri<=0;end
            if(operand_valid && operand_ready) begin
                if({1'b0,ri}==counts[read_bank]-1'b1) begin
                    full[read_bank]<=0;read_bank<=!read_bank;reading<=0;ri<=0;
                end else ri<=ri+1'b1;
            end
        end
    end
endmodule
`default_nettype wire
