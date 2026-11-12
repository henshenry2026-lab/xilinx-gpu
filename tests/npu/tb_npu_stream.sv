`timescale 1ns/1ps
module tb_npu_stream #(
    parameter integer LANES=4, K_TILE=16, LINEAR=0
);
    reg clk=0; always #5 clk=~clk;
    reg rst=1,cmd_valid=0,cmd_relu=0;
    wire cmd_ready,busy,done,error;
    reg [15:0] cmd_m=0,cmd_n=0,cmd_k=0;
    reg operand_valid=0; wire operand_ready;
    reg [LANES*8-1:0] a_vector=0,b_vector=0;
    reg param_valid=0; wire param_ready;
    reg signed [31:0] param_bias=0,param_multiplier=1;
    reg [5:0] param_shift=0;
    wire out_valid,out_last; reg out_ready=0,random_ready=0;
    wire [15:0] out_row,out_col;
    wire signed [31:0] result;
    wire [63:0] stat_cycles,stat_load_cycles,stat_compute_cycles,stat_input_wait_cycles;
    wire [63:0] stat_buffer_stall_cycles,stat_output_stall_cycles,stat_overlap_cycles,stat_mac_count;
    generate if(LINEAR) begin: linear_path
        wire signed [7:0] out_data;
        npu_stream_linear_top #(.LANES(LANES),.K_TILE(K_TILE)) dut(.*);
        assign result={{24{out_data[7]}},out_data};
    end else begin: raw_path
        wire signed [31:0] out_data;
        npu_stream_top #(.LANES(LANES),.K_TILE(K_TILE)) dut (
            .clk(clk),.rst(rst),.cmd_valid(cmd_valid),.cmd_ready(cmd_ready),
            .cmd_m(cmd_m),.cmd_n(cmd_n),.cmd_k(cmd_k),.operand_valid(operand_valid),
            .operand_ready(operand_ready),.a_vector(a_vector),.b_vector(b_vector),
            .busy(busy),.done(done),.error(error),.out_valid(out_valid),.out_ready(out_ready),
            .out_row(out_row),.out_col(out_col),.out_data(out_data),.out_last(out_last),
            .stat_cycles(stat_cycles),.stat_load_cycles(stat_load_cycles),
            .stat_compute_cycles(stat_compute_cycles),.stat_input_wait_cycles(stat_input_wait_cycles),
            .stat_buffer_stall_cycles(stat_buffer_stall_cycles),
            .stat_output_stall_cycles(stat_output_stall_cycles),
            .stat_overlap_cycles(stat_overlap_cycles),.stat_mac_count(stat_mac_count));
        assign result=out_data;
        assign param_ready=1;
    end endgenerate
    integer a[0:4095],b[0:16383],raw[0:255],quantized[0:255];
    integer bias[0:31],multiplier[0:31],shift[0:31],mlp_hidden[0:31];
    reg seen[0:255];
    integer fd,report,count,case_index,m,n,depth,relu,mlp,r,c,k,value,outputs,lasts;
    integer tr,tc,rr,cc,vector_count,chunk_count,last_stall,overlapped=0;
    reg checking=0,stalled=0;
    reg [64:0] held;
    reg [31:0] rng=32'h397216af;
    always @(negedge clk) begin
        rng={rng[30:0],rng[31]^rng[21]^rng[1]^rng[0]};
        if(random_ready) begin
            if(out_valid&&out_last&&last_stall<12) begin out_ready=0;last_stall=last_stall+1;end
            else out_ready=rng[0]|rng[5];
        end
    end
    always @(posedge clk) begin
        if(rst) stalled=0;
        else begin
            if(stalled&&(!out_valid||{out_last,out_row,out_col,result}!==held))
                $fatal(1,"stream output changed under backpressure");
            stalled=out_valid&&!out_ready;
            held={out_last,out_row,out_col,result};
            if(checking&&out_valid&&out_ready) begin
                if(out_row>=m||out_col>=n||seen[out_row*32+out_col]) $fatal(1,"bad/duplicate coordinate");
                seen[out_row*32+out_col]=1;
                if(result!==(LINEAR?quantized[out_row*32+out_col]:raw[out_row*32+out_col]))
                    $fatal(1,"case %0d [%0d,%0d] expected %0d got %0d",case_index,out_row,out_col,
                           LINEAR?quantized[out_row*32+out_col]:raw[out_row*32+out_col],result);
                if(out_last!==(out_row==m-1&&out_col==n-1)) $fatal(1,"last mismatch");
                if(LINEAR&&mlp==1) mlp_hidden[out_col]=result;
                outputs=outputs+1; if(out_last) lasts=lasts+1;
            end
        end
    end
    task read_int(output integer v);
        if($fscanf(fd,"%d",v)!=1) $fatal(1,"invalid stream vectors");
    endtask
    task launch(input integer rows,cols,inner);
        begin
            @(negedge clk);cmd_m=rows;cmd_n=cols;cmd_k=inner;cmd_valid=1;
            do @(posedge clk); while(!cmd_ready);
            @(negedge clk);cmd_valid=0;
        end
    endtask
    task feed_vector;
        begin
            operand_valid=1;
            do @(posedge clk); while(!operand_ready);
            @(negedge clk);operand_valid=0;
        end
    endtask
    task reset_job;
        begin
            @(negedge clk);rst=1;
            @(negedge clk);rst=0;operand_valid=0;param_valid=0;out_ready=0;
            #1;if(busy||out_valid||done||error||!cmd_ready) $fatal(1,"stream reset did not abort");
        end
    endtask
    task send_operands;
        begin
            for(tr=0;tr<m;tr=tr+LANES) for(tc=0;tc<n;tc=tc+LANES)
                for(k=0;k<depth;k=k+1) begin
                    if((k+tc)%11==3) repeat(4) @(negedge clk);
                    for(rr=0;rr<LANES;rr=rr+1)
                        a_vector[rr*8+:8]=tr+rr<m?a[(tr+rr)*512+k]:0;
                    for(cc=0;cc<LANES;cc=cc+1)
                        b_vector[cc*8+:8]=tc+cc<n?b[k*32+tc+cc]:0;
                    feed_vector();
                end
        end
    endtask
    task send_parameters;
        integer pr,pc,lr,lc,col;
        begin
            if(LINEAR) begin
                // Invalid parameters are reported without consuming the pending raw result.
                for(col=0;col<3;col=col+1) begin
                    param_multiplier=col==0?0:col==1?-1:1;
                    param_shift=col==2?32:0;param_valid=1;
                    do @(posedge clk); while(!param_ready);
                    #1;if(!error) $fatal(1,"invalid scale not reported");
                    @(negedge clk);param_valid=0;
                end
                for(pr=0;pr<m;pr=pr+LANES) for(pc=0;pc<n;pc=pc+LANES)
                    for(lr=0;lr<LANES&&pr+lr<m;lr=lr+1)
                        for(lc=0;lc<LANES&&pc+lc<n;lc=lc+1) begin
                            col=pc+lc;
                            if(col%3==1) repeat(3) @(negedge clk);
                            param_bias=bias[col];param_multiplier=multiplier[col];param_shift=shift[col];
                            param_valid=1;
                            do @(posedge clk); while(!param_ready);
                            @(negedge clk);param_valid=0;
                        end
            end
        end
    endtask
    initial begin
        repeat(4) @(negedge clk);rst=0;
        launch(0,1,1);if(!error||busy) $fatal(1,"zero shape accepted");
        launch(1,1,0);if(!error||busy) $fatal(1,"zero K accepted");
        // Reset while waiting for a block, after partial loading, computing, and stalled output.
        launch(1,1,17);reset_job();
        launch(1,1,17);feed_vector();reset_job();
        launch(1,1,17);repeat(16) feed_vector();repeat(3) @(negedge clk);reset_job();
        launch(1,1,1);param_valid=1;param_multiplier=1;param_shift=0;
        feed_vector();wait(out_valid);repeat(6) @(negedge clk);reset_job();
        fd=$fopen("build/npu_stream_cases.txt","r");if(!fd)$fatal(1,"stream vectors missing");
        if(LINEAR) report=$fopen($sformatf("build/npu-stream-linear-%0d-cycles.txt",LANES),"w");
        else report=$fopen($sformatf("build/npu-stream-raw-%0d-cycles.txt",LANES),"w");
        if(!report)$fatal(1,"measurement file unavailable");
        read_int(count);
        for(case_index=0;case_index<count;case_index=case_index+1) begin
            read_int(m);read_int(n);read_int(depth);read_int(relu);read_int(mlp);
            for(r=0;r<m;r=r+1) for(k=0;k<depth;k=k+1) begin
                read_int(value);a[r*512+k]=LINEAR&&mlp==2?mlp_hidden[k]:value;
            end
            for(k=0;k<depth;k=k+1) for(c=0;c<n;c=c+1) read_int(b[k*32+c]);
            for(c=0;c<n;c=c+1) begin read_int(bias[c]);read_int(multiplier[c]);read_int(shift[c]);end
            for(r=0;r<m;r=r+1) for(c=0;c<n;c=c+1) read_int(raw[r*32+c]);
            for(r=0;r<m;r=r+1) for(c=0;c<n;c=c+1) read_int(quantized[r*32+c]);
            for(r=0;r<256;r=r+1) seen[r]=0;
            outputs=0;lasts=0;last_stall=0;checking=1;cmd_relu=relu;
            launch(m,n,depth);
            cmd_relu=!relu; // Configuration must be latched at the accepted command.
            random_ready=1;
            fork send_operands();send_parameters(); join
            while(!done) begin
                if(!busy||cmd_ready) $fatal(1,"input window reopened before job done");
                @(negedge clk);
            end
            @(negedge clk);random_ready=0;out_ready=0;checking=0;
            if(outputs!=m*n||lasts!=1||busy||error) $fatal(1,"stream completion/count mismatch");
            vector_count=((m+LANES-1)/LANES)*((n+LANES-1)/LANES)*depth;
            chunk_count=((m+LANES-1)/LANES)*((n+LANES-1)/LANES)*((depth+K_TILE-1)/K_TILE);
            if(stat_load_cycles!=vector_count||stat_mac_count!=m*n*depth)
                $fatal(1,"stream transfer/MAC counter mismatch");
            if(stat_compute_cycles!=vector_count+chunk_count*(2*(LANES-1)+1))
                $fatal(1,"stream compute cycle model mismatch");
            if(stat_cycles<stat_compute_cycles||stat_cycles<stat_load_cycles)
                $fatal(1,"invalid lower bound");
            if(stat_overlap_cycles>0) overlapped=1;
            $fdisplay(report,"%0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",m,n,depth,
                stat_cycles,stat_load_cycles,stat_compute_cycles,stat_input_wait_cycles,
                stat_buffer_stall_cycles,stat_output_stall_cycles,stat_overlap_cycles,stat_mac_count,mlp);
            @(negedge clk);if(done) $fatal(1,"done not a pulse");
        end
        if(K_TILE>1&&!overlapped)$fatal(1,"no actual load/compute overlap observed");
        $fclose(fd);$fclose(report);
        $display("PASS tb_npu_stream LANES=%0d LINEAR=%0d: %0d matrices incl live MLP, overlap/reset/stalls",LANES,LINEAR,count);
        $finish;
    end
    initial begin #30000000;$fatal(1,"stream timeout");end
endmodule
