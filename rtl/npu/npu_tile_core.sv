`timescale 1ns/1ps
`default_nettype none
// One output tile, with partial sums retained across explicitly delimited K chunks.
module npu_tile_core #(
    parameter integer LANES=4,
    parameter integer K_TILE=16
) (
    input wire clk,rst,cmd_valid,
    output wire cmd_ready,
    input wire [15:0] base_row,base_col,rows_valid,cols_valid,k_count,
    input wire clear_accum,emit_result,job_last,
    input wire operand_valid,
    output wire operand_ready,
    input wire [LANES*8-1:0] a_vector,b_vector,
    input wire [LANES-1:0] a_mask,b_mask,
    output wire busy,
    output reg done,error,
    output wire out_valid,
    input wire out_ready,
    output wire [15:0] out_row,out_col,
    output wire signed [31:0] out_data,
    output wire out_last,
    output wire compute_active,phase_active
);
    localparam IDLE=0,SETUP=1,FEED=2,DRAIN=3,FINISH=4,EMIT=5;
    reg [2:0] state;
    reg [15:0] row_base,col_base,rows,cols,count,received,drained,er,ec,total_k;
    reg first_chunk,emit_chunk,last_job,context_valid;
    wire shape_valid=rows_valid!=0 && rows_valid<=LANES && cols_valid!=0 && cols_valid<=LANES
        && k_count!=0 && k_count<=K_TILE
        && {1'b0,base_row}+{1'b0,rows_valid}<=17'd65536
        && {1'b0,base_col}+{1'b0,cols_valid}<=17'd65536 && (!job_last||emit_result);
    wire continuation_valid=context_valid && base_row==row_base && base_col==col_base
        && rows_valid==rows && cols_valid==cols
        && {1'b0,total_k}+{1'b0,k_count}<=17'd65535;
    wire [LANES*LANES*32-1:0] sums;
    reg [LANES-1:0] av,bv;
    integer lane;
    assign cmd_ready=!rst && state==IDLE;
    assign operand_ready=!rst && state==FEED;
    assign busy=state!=IDLE;
    assign compute_active=(state==FEED && operand_valid) || state==DRAIN;
    assign phase_active=state==SETUP || compute_active;
    assign out_valid=!rst && state==EMIT;
    assign out_row=row_base+er;
    assign out_col=col_base+ec;
    assign out_data=sums[(er*LANES+ec)*32+:32];
    assign out_last=out_valid && last_job && er==rows-1'b1 && ec==cols-1'b1;
    always @* begin
        av=0; bv=0;
        for(lane=0;lane<LANES;lane=lane+1) begin
            av[lane]=state==FEED && operand_valid && a_mask[lane] && lane<rows;
            bv[lane]=state==FEED && operand_valid && b_mask[lane] && lane<cols;
        end
    end
    npu_systolic_array #(.LANES(LANES)) array (
        .clk(clk),.rst(rst),.clear(state==SETUP && first_chunk),.step(compute_active),
        .a_data(a_vector),.b_data(b_vector),.a_valid(av),.b_valid(bv),.sums(sums)
    );
    always @(posedge clk) begin
        if(rst) begin
            state<=IDLE; done<=0; error<=0; context_valid<=0;
            row_base<=0; col_base<=0; rows<=0; cols<=0; count<=0; total_k<=0;
            received<=0; drained<=0; er<=0; ec<=0;
            first_chunk<=0; emit_chunk<=0; last_job<=0;
        end else begin
            done<=0; error<=0;
            case(state)
                IDLE: if(cmd_valid) begin
                    if(!shape_valid || (!clear_accum && !continuation_valid)) error<=1;
                    else begin
                        row_base<=base_row; col_base<=base_col; rows<=rows_valid; cols<=cols_valid;
                        count<=k_count; total_k<=clear_accum?k_count:total_k+k_count;
                        first_chunk<=clear_accum; emit_chunk<=emit_result; last_job<=job_last;
                        received<=0; drained<=0; er<=0; ec<=0; state<=SETUP;
                    end
                end
                SETUP: state<=FEED;
                FEED: if(operand_valid) begin
                    if(received==count-1'b1) state<=LANES==1?FINISH:DRAIN;
                    else received<=received+1'b1;
                end
                DRAIN: if(drained==2*(LANES-1)-1) state<=FINISH;
                       else drained<=drained+1'b1;
                FINISH: begin
                    context_valid<=!emit_chunk;
                    if(emit_chunk) state<=EMIT;
                    else begin state<=IDLE; done<=1; end
                end
                EMIT: if(out_ready) begin
                    if(ec!=cols-1'b1) ec<=ec+1'b1;
                    else if(er!=rows-1'b1) begin ec<=0; er<=er+1'b1; end
                    else begin state<=IDLE; done<=1; end
                end
                default: state<=IDLE;
            endcase
        end
    end
endmodule
`default_nettype wire
