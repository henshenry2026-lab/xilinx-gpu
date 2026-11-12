`timescale 1ns/1ps
// Complete 16-bit dimension boundaries, rather than merely accepting descriptors.
module tb_npu_stream_edges;
    localparam LANES=2,K_TILE=3;
    reg clk=0;always #5 clk=~clk;
    reg rst=1,cmd_valid=0,operand_valid=0,out_ready=1;
    reg [15:0] cmd_m=0,cmd_n=0,cmd_k=0;
    reg [15:0] a_vector=16'h8080,b_vector=16'h8080;
    wire cmd_ready,operand_ready,busy,done,error,out_valid,out_last;
    wire [15:0] out_row,out_col;
    wire signed [31:0] out_data;
    wire [63:0] stat_cycles,stat_load_cycles,stat_compute_cycles,stat_input_wait_cycles;
    wire [63:0] stat_buffer_stall_cycles,stat_output_stall_cycles,stat_overlap_cycles,stat_mac_count;
    npu_stream_top #(.LANES(LANES),.K_TILE(K_TILE)) dut(.*);
    integer outputs,lasts,expected,m,n,k,er,ec,tile_row,tile_col;
    always @(posedge clk) if(!rst && out_valid && out_ready) begin
        if(out_row!=er||out_col!=ec||out_data!==expected) $fatal(1,"16-bit edge data/coordinate mismatch");
        if(out_last!==(outputs==m*n-1)) $fatal(1,"16-bit edge last mismatch");
        outputs=outputs+1;lasts=lasts+out_last;
        if(ec+1<tile_col+LANES && ec+1<n) ec=ec+1;
        else if(er+1<tile_row+LANES && er+1<m) begin er=er+1;ec=tile_col;end
        else if(tile_col+LANES<n) begin tile_col=tile_col+LANES;er=tile_row;ec=tile_col;end
        else begin tile_col=0;tile_row=tile_row+LANES;er=tile_row;ec=0;end
    end
    task run_job(input integer rows,cols,depth);
        integer v,vectors;
        begin
            m=rows;n=cols;k=depth;expected=depth*16384;
            outputs=0;lasts=0;er=0;ec=0;tile_row=0;tile_col=0;
            @(negedge clk);cmd_m=rows;cmd_n=cols;cmd_k=depth;cmd_valid=1;
            do @(posedge clk);while(!cmd_ready);
            @(negedge clk);cmd_valid=0;
            vectors=((rows+LANES-1)/LANES)*((cols+LANES-1)/LANES)*depth;
            operand_valid=1;
            for(v=0;v<vectors;v=v+1) begin
                do @(posedge clk);while(!operand_ready);
                @(negedge clk);
            end
            operand_valid=0;
            // A pending next descriptor must not be accepted while draining output.
            cmd_valid=1;cmd_m=1;cmd_n=1;cmd_k=1;
            while(!done) begin
                if(!busy||cmd_ready) $fatal(1,"edge input window reopened early");
                @(negedge clk);
            end
            cmd_valid=0;
            if(outputs!=rows*cols||lasts!=1||stat_mac_count!=rows*cols*depth||error)
                $fatal(1,"16-bit edge counts mismatch");
            @(negedge clk);
        end
    endtask
    initial begin
        repeat(3) @(negedge clk);rst=0;
        run_job(1,1,65535);
        run_job(1,65535,1);
        run_job(65535,1,1);
        run_job(1,1,1);
        $display("PASS tb_npu_stream_edges: full K/M/N=65535, non-power-of-two chunks, pending command");
        $finish;
    end
    initial begin #30000000;$fatal(1,"16-bit edge timeout");end
endmodule
