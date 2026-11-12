`timescale 1ns/1ps
module tb_npu_tile_core;
    localparam LANES=4, K_TILE=16;
    reg clk=0; always #5 clk=~clk;
    reg rst=1, cmd_valid=0, clear_accum=1, emit_result=0, job_last=0;
    wire cmd_ready, busy, done, error, compute_active, phase_active;
    reg [15:0] base_row=3, base_col=5, rows_valid=3, cols_valid=2, k_count=1;
    reg operand_valid=0; wire operand_ready;
    reg [31:0] a_vector=0,b_vector=0;
    reg [3:0] a_mask=15,b_mask=15;
    wire out_valid, out_last; reg out_ready=0;
    wire [15:0] out_row,out_col;
    wire signed [31:0] out_data;
    npu_tile_core dut(.*);
    task launch(input integer count, input bit first, final_chunk);
        begin
            @(negedge clk); k_count=count; clear_accum=first;
            emit_result=final_chunk; job_last=final_chunk; cmd_valid=1;
            @(negedge clk); cmd_valid=0;
        end
    endtask
    integer k,r,c;
    initial begin
        repeat(3) @(negedge clk); rst=0;
        launch(0,1,0);
        if(!error||busy) $fatal(1,"bad chunk accepted");
        launch(17,1,0);
        if(!error||busy) $fatal(1,"oversize chunk accepted");
        launch(1,0,1);
        if(!error||busy) $fatal(1,"continuation without context accepted");
        for(k=0;k<2;k=k+1) begin
            launch(1,k==0,k==1);
            wait(operand_ready);
            @(negedge clk); operand_valid=1;
            for(r=0;r<4;r=r+1) a_vector[r*8+:8]=r-2;
            for(c=0;c<4;c=c+1) b_vector[c*8+:8]=c+3;
            @(negedge clk); operand_valid=0;
            if(k==0) begin
                wait(done); @(negedge clk);
                base_row=4;launch(1,0,1);
                if(!error||busy) $fatal(1,"continuation changed tile context");
                base_row=3;
            end
        end
        wait(out_valid); repeat(5) @(negedge clk);
        for(r=0;r<3;r=r+1) for(c=0;c<2;c=c+1) begin
            if(!out_valid||out_row!=r+3||out_col!=c+5||out_data!==2*(r-2)*(c+3))
                $fatal(1,"tile accumulation/coordinates mismatch");
            if(out_last!==(r==2&&c==1)) $fatal(1,"last mismatch");
            out_ready=1; @(negedge clk);
        end
        out_ready=0;
        wait(done); @(negedge clk);
        if(busy) $fatal(1,"tile still busy after done");
        // Stalled input must freeze all skew/PE state; masks suppress selected rows/columns.
        a_mask=4'b1101;b_mask=4'b1110;
        launch(2,1,1);wait(operand_ready);
        for(k=0;k<2;k=k+1) begin
            repeat(3) @(negedge clk);
            operand_valid=1;
            @(negedge clk);operand_valid=0;
        end
        wait(out_valid);@(negedge clk);
        for(r=0;r<3;r=r+1) for(c=0;c<2;c=c+1) begin
            if(out_data!==((r==1||c==0)?0:2*(r-2)*(c+3)))
                $fatal(1,"tile mask/stalled input [%0d,%0d] got %0d expected %0d",r,c,out_data,
                       (r==1||c==0)?0:2*(r-2)*(c+3));
            out_ready=1;@(negedge clk);
        end
        out_ready=0;wait(done);@(negedge clk);
        $display("PASS tb_npu_tile_core: invalid/context chunks, retained partial sums, masks and stalls"); $finish;
    end
    initial begin #100000; $fatal(1,"tile timeout"); end
endmodule
