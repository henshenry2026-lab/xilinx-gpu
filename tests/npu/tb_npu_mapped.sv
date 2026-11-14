`timescale 1ns/1ps
// Consume mapper-produced streams. The next layer uses captured RTL activations;
// compare exported A bytes against those activations before every transfer.
module tb_npu_mapped #(
    parameter integer LANES=4,K_TILE=16,BACKEND=0,STALLS=1
);
    localparam MAX_BATCH=8,MAX_FEATURES=256;
    reg clk=0;always #5 clk=~clk;
    reg rst=1,cmd_valid=0,cmd_relu=0,operand_valid=0,param_valid=0;
    wire cmd_ready,operand_ready,param_ready,busy,done,error;
    reg [15:0] cmd_m=0,cmd_n=0,cmd_k=0;
    reg [LANES*8-1:0] a_vector=0,b_vector=0;
    reg signed [7:0] a_scalar=0;
    reg signed [31:0] param_bias=0,param_multiplier=1;
    reg [5:0] param_shift=0;
    wire out_valid,out_last;reg out_ready=0;
    wire [15:0] out_row,out_col;
    wire signed [7:0] out_data;
    wire [63:0] stat_cycles,stat_load_cycles,stat_compute_cycles,stat_input_wait_cycles;
    wire [63:0] stat_buffer_stall_cycles,stat_output_stall_cycles,stat_overlap_cycles,stat_mac_count;
    generate if(BACKEND==0) begin: gemm
        npu_stream_linear_top #(.LANES(LANES),.K_TILE(K_TILE)) dut(.*);
    end else begin: gemv
        npu_gemv_linear_top #(.LANES(LANES)) dut(.*);
    end endgenerate
    integer activations[0:MAX_BATCH*MAX_FEATURES-1],next_activations[0:MAX_BATCH*MAX_FEATURES-1];
    integer expected[0:MAX_BATCH*MAX_FEATURES-1];
    reg seen[0:MAX_BATCH*MAX_FEATURES-1];
    integer jobs_fd,input_fd,operand_fd,param_fd,expected_fd,results_fd,cycles_fd;
    integer job_count,file_lanes,file_chunk,job,m,n,k,relu,r,c,v,outputs,lasts,last_hold;
    integer file_value;
    integer expected_row,expected_col;
    string directory,backend_marker;
    reg checking=0,stalled=0;
    reg [40:0] held;
    reg [31:0] rng=32'h418c97ab;
    always @(negedge clk) begin
        rng={rng[30:0],rng[31]^rng[21]^rng[1]^rng[0]};
        if(checking) begin
            if(!STALLS) out_ready=1;
            else if(out_valid&&out_last&&last_hold<8) begin out_ready=0;last_hold=last_hold+1;end
            else out_ready=rng[0]|rng[3];
        end else out_ready=0;
    end
    always @(posedge clk) if(!rst) begin
        if(error) $fatal(1,"mapped descriptor or parameter rejected");
        if(stalled&&(!out_valid||{out_last,out_row,out_col,out_data}!==held))
            $fatal(1,"mapped output changed while stalled");
        stalled=out_valid&&!out_ready;held={out_last,out_row,out_col,out_data};
        if(checking&&out_valid&&out_ready) begin
            if(out_row>=m||out_col>=n||seen[out_row*MAX_FEATURES+out_col])
                $fatal(1,"mapped invalid/duplicate output coordinate");
            if(out_row!=expected_row||out_col!=expected_col) $fatal(1,"mapped output order invalid");
            if($signed(out_data)!==expected[out_row*MAX_FEATURES+out_col])
                $fatal(1,"mapped layer %0d [%0d,%0d] got %0d expected %0d",job,out_row,out_col,
                       $signed(out_data),expected[out_row*MAX_FEATURES+out_col]);
            if(out_last!==(outputs==m*n-1)) $fatal(1,"mapped last mismatch");
            seen[out_row*MAX_FEATURES+out_col]=1;
            next_activations[out_row*MAX_FEATURES+out_col]=$signed(out_data);
            $fdisplay(results_fd,"%0d %0d %0d %0d",job,out_row,out_col,$signed(out_data));
            outputs=outputs+1;lasts=lasts+out_last;
            if(expected_col+1<n&&(expected_col+1)%LANES!=0) expected_col=expected_col+1;
            else if(expected_row+1<m&&(expected_row+1)%LANES!=0) begin
                expected_row=expected_row+1;expected_col=(expected_col/LANES)*LANES;
            end else if((expected_col/LANES+1)*LANES<n) begin
                expected_row=(expected_row/LANES)*LANES;expected_col=(expected_col/LANES+1)*LANES;
            end else begin expected_row=(expected_row/LANES+1)*LANES;expected_col=0;end
        end
    end
    function automatic integer valid_hex(input string token,input integer digits);
        integer index;
        reg [7:0] character;
        begin
            valid_hex=token.len()==digits;
            for(index=0;index<token.len();index=index+1) begin
                character=token[index];
                if(!((character>="0"&&character<="9")||(character>="a"&&character<="f")
                    ||(character>="A"&&character<="F"))) valid_hex=0;
            end
        end
    endfunction
    // Validate the entire token before any narrowing conversion. Ten decimal
    // digits cover INT32; the pre-multiply limit check also prevents wrap.
    task automatic read_decimal(input integer fd,input signed [63:0] minimum,maximum,
                                input string label_text,output integer value);
        string token;
        integer first,index,digit;
        reg negative;
        reg [63:0] magnitude,limit;
        reg signed [63:0] parsed;
        begin
            if($fscanf(fd,"%s",token)!=1) $fatal(1,"mapped %s decimal token missing",label_text);
            first=0;negative=0;magnitude=0;
            if(token.len()>0&&(token[0]=="-"||token[0]=="+")) begin
                negative=token[0]=="-";first=1;
            end
            if(token.len()-first<1||token.len()-first>10)
                $fatal(1,"mapped %s decimal encoding/range invalid",label_text);
            if(negative&&minimum>=0) $fatal(1,"mapped %s decimal range invalid",label_text);
            limit=negative?-minimum:maximum;
            for(index=first;index<token.len();index=index+1) begin
                if(token[index]<"0"||token[index]>"9")
                    $fatal(1,"mapped %s decimal encoding invalid",label_text);
                digit=token[index]-"0";
                if(magnitude>limit/10||(magnitude==limit/10&&digit>limit%10))
                    $fatal(1,"mapped %s decimal range invalid",label_text);
                magnitude=magnitude*10+digit;
            end
            parsed=negative?-$signed(magnitude):$signed(magnitude);
            if(parsed<minimum||parsed>maximum) $fatal(1,"mapped %s decimal range invalid",label_text);
            value=parsed;
        end
    endtask
    task send_operands;
        integer tr,tc,inner,lane,actual;
        reg [LANES*8-1:0] mapped_a;
        reg [7:0] mapped_scalar;
        string a_token,b_token;
        begin
            for(tr=0;tr<m;tr=tr+LANES) for(tc=0;tc<n;tc=tc+LANES)
                for(inner=0;inner<k;inner=inner+1) begin
                    if($fscanf(operand_fd,"%s %s",a_token,b_token)!=2) $fatal(1,"mapped operands truncated");
                    if(!valid_hex(a_token,BACKEND==1?2:2*LANES)||!valid_hex(b_token,2*LANES))
                        $fatal(1,"mapped operand encoding invalid");
                    if($sscanf(b_token,"%h",b_vector)!=1) $fatal(1,"mapped operand encoding invalid");
                    if(BACKEND==1) begin
                        if($sscanf(a_token,"%h",mapped_scalar)!=1) $fatal(1,"mapped operand encoding invalid");
                        actual=activations[inner];
                        if(mapped_scalar!==actual[7:0])
                            $fatal(1,"mapped A differs from actual RTL activation at layer %0d",job);
                        a_scalar=actual[7:0];
                    end else begin
                        if($sscanf(a_token,"%h",mapped_a)!=1) $fatal(1,"mapped operand encoding invalid");
                        for(lane=0;lane<LANES;lane=lane+1) begin
                            actual=tr+lane<m?activations[(tr+lane)*MAX_FEATURES+inner]:0;
                            if(mapped_a[lane*8+:8]!==actual[7:0])
                                $fatal(1,"mapped A differs from actual RTL activation at layer %0d",job);
                            a_vector[lane*8+:8]=actual[7:0];
                        end
                    end
                    if(STALLS&&(inner+tc)%7==2) repeat(3) @(negedge clk);
                    operand_valid=1;
                    do @(posedge clk);while(!operand_ready);
                    @(negedge clk);operand_valid=0;
                end
        end
    endtask
    task send_parameters;
        integer tr,tc,lr,lc,pr,pc,pb,pm,ps;
        begin
            for(tr=0;tr<m;tr=tr+LANES) for(tc=0;tc<n;tc=tc+LANES)
                for(lr=tr;lr<tr+LANES&&lr<m;lr=lr+1)
                    for(lc=tc;lc<tc+LANES&&lc<n;lc=lc+1) begin
                        read_decimal(param_fd,0,m-1,"parameter row",pr);
                        read_decimal(param_fd,0,n-1,"parameter column",pc);
                        read_decimal(param_fd,-64'sd2147483648,64'sd2147483647,"parameter bias",pb);
                        read_decimal(param_fd,1,64'sd2147483647,"parameter multiplier",pm);
                        read_decimal(param_fd,0,31,"parameter shift",ps);
                        if(pr!=lr||pc!=lc) $fatal(1,"mapped parameter order/value invalid");
                        param_bias=pb;param_multiplier=pm;param_shift=ps;
                        if(STALLS&&lc%5==1) repeat(2) @(negedge clk);
                        param_valid=1;
                        do @(posedge clk);while(!param_ready);
                        @(negedge clk);param_valid=0;
                    end
        end
    endtask
    task require_eof(input integer fd);
        integer scanned;
        reg [1023:0] item;
        begin
            scanned=$fscanf(fd,"%s",item);
            // Icarus may return zero after consuming trailing whitespace at EOF.
            if(scanned>0) $fatal(1,"mapped stream has extra tokens: %s",item);
        end
    endtask
    initial begin
        if(!$value$plusargs("BUNDLE=%s",directory)) $fatal(1,"pass +BUNDLE=directory");
        jobs_fd=$fopen({directory,"/jobs.txt"},"r");
        input_fd=$fopen({directory,"/inputs.txt"},"r");
        operand_fd=$fopen({directory,"/operands.txt"},"r");
        param_fd=$fopen({directory,"/parameters.txt"},"r");
        expected_fd=$fopen({directory,"/expected.txt"},"r");
        results_fd=$fopen({directory,"/rtl-results.txt"},"w");
        cycles_fd=$fopen({directory,"/rtl-cycles.txt"},"w");
        if(!jobs_fd||!input_fd||!operand_fd||!param_fd||!expected_fd||!results_fd||!cycles_fd)
            $fatal(1,"mapped bundle file unavailable");
        if(BACKEND!=0&&BACKEND!=1) $fatal(1,"mapped backend invalid");
        if(BACKEND==1) begin
            if($fscanf(jobs_fd,"%s",backend_marker)!=1||backend_marker!="gemv-v1")
                $fatal(1,"mapped GEMV header/config invalid");
            read_decimal(jobs_fd,1,64,"layer count",job_count);
            read_decimal(jobs_fd,1,16,"header lanes",file_lanes);
            if(file_lanes!=LANES) $fatal(1,"mapped GEMV header/config invalid");
        end else begin
            read_decimal(jobs_fd,1,64,"layer count",job_count);
            read_decimal(jobs_fd,1,16,"header lanes",file_lanes);
            read_decimal(jobs_fd,1,65535,"header k_tile",file_chunk);
            if(file_lanes!=LANES||file_chunk!=K_TILE) $fatal(1,"mapped header/config invalid");
        end
        repeat(4) @(negedge clk);rst=0;
        for(job=0;job<job_count;job=job+1) begin
            read_decimal(jobs_fd,1,BACKEND==1?1:MAX_BATCH,"shape m",m);
            read_decimal(jobs_fd,1,MAX_FEATURES,"shape n",n);
            read_decimal(jobs_fd,1,MAX_FEATURES,"shape k",k);
            read_decimal(jobs_fd,0,1,"relu",relu);
            for(r=0;r<m;r=r+1) for(c=0;c<n;c=c+1) begin
                read_decimal(expected_fd,-128,127,"expected output",file_value);
                expected[r*MAX_FEATURES+c]=file_value;seen[r*MAX_FEATURES+c]=0;
            end
            if(job==0) begin
                for(r=0;r<m;r=r+1) for(c=0;c<k;c=c+1) begin
                    read_decimal(input_fd,-128,127,"input",file_value);
                    activations[r*MAX_FEATURES+c]=file_value;
                end
            end
            outputs=0;lasts=0;last_hold=0;expected_row=0;expected_col=0;checking=1;
            @(negedge clk);cmd_m=m;cmd_n=n;cmd_k=k;cmd_relu=relu;cmd_valid=1;
            do @(posedge clk);while(!cmd_ready);
            @(negedge clk);cmd_valid=0;
            fork send_operands();send_parameters();join
            while(!done) begin
                if(!busy||cmd_ready) $fatal(1,"mapped command window reopened before done");
                @(negedge clk);
            end
            @(negedge clk);checking=0;
            if(outputs!=m*n||lasts!=1||busy) $fatal(1,"mapped output count/completion mismatch");
            $fdisplay(cycles_fd,"%0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",job,m,n,k,
                stat_cycles,stat_load_cycles,stat_compute_cycles,stat_input_wait_cycles,
                stat_buffer_stall_cycles,stat_output_stall_cycles,stat_overlap_cycles,stat_mac_count);
            for(r=0;r<m;r=r+1) for(c=0;c<n;c=c+1)
                activations[r*MAX_FEATURES+c]=next_activations[r*MAX_FEATURES+c];
            @(negedge clk);if(done) $fatal(1,"mapped done did not pulse");
        end
        require_eof(jobs_fd);require_eof(input_fd);require_eof(operand_fd);
        require_eof(param_fd);require_eof(expected_fd);
        $fclose(results_fd);$fclose(cycles_fd);
        $display("PASS tb_npu_mapped BACKEND=%0d LANES=%0d K_TILE=%0d STALLS=%0d: %0d mapped layers, live activations",BACKEND,LANES,K_TILE,STALLS,job_count);
        $finish;
    end
    initial begin #100000000;$fatal(1,"mapped simulation timeout");end
endmodule
