`timescale 1ns/1ps
`default_nettype none
// Standalone, deterministic interface tests; no mapper-generated files required.
module tb_npu_gemv #(
    parameter integer LANES=4,
    parameter integer LINEAR=0
);
    reg clk=0;
    always #5 clk=~clk;
    reg rst=1,cmd_valid=0,operand_valid=0,param_valid=0,out_ready=0,cmd_relu=0;
    reg [15:0] cmd_n=0,cmd_k=0;
    reg signed [7:0] a_scalar=0;
    reg [LANES*8-1:0] b_vector=0;
    reg signed [31:0] param_bias=0,param_multiplier=1;
    reg [5:0] param_shift=0;
    wire cmd_ready,operand_ready,param_ready,busy,done,error,out_valid,out_last;
    wire [15:0] out_row,out_col;
    wire signed [31:0] out_data;
    wire [63:0] stat_cycles,stat_load_cycles,stat_compute_cycles;
    wire [63:0] stat_input_wait_cycles,stat_buffer_stall_cycles;
    wire [63:0] stat_output_stall_cycles,stat_overlap_cycles,stat_mac_count;
    generate if(LINEAR) begin: linear_dut
        wire signed [7:0] q;
        assign out_data={{24{q[7]}},q};
        npu_gemv_linear_top #(.LANES(LANES)) dut (
            .clk(clk),.rst(rst),.cmd_valid(cmd_valid),.cmd_ready(cmd_ready),
            .cmd_n(cmd_n),.cmd_k(cmd_k),.cmd_relu(cmd_relu),
            .operand_valid(operand_valid),.operand_ready(operand_ready),
            .a_scalar(a_scalar),.b_vector(b_vector),.param_valid(param_valid),
            .param_ready(param_ready),.param_bias(param_bias),
            .param_multiplier(param_multiplier),.param_shift(param_shift),
            .busy(busy),.done(done),.error(error),.out_valid(out_valid),
            .out_ready(out_ready),.out_row(out_row),.out_col(out_col),
            .out_data(q),.out_last(out_last),
            .stat_cycles(stat_cycles),.stat_load_cycles(stat_load_cycles),
            .stat_compute_cycles(stat_compute_cycles),
            .stat_input_wait_cycles(stat_input_wait_cycles),
            .stat_buffer_stall_cycles(stat_buffer_stall_cycles),
            .stat_output_stall_cycles(stat_output_stall_cycles),
            .stat_overlap_cycles(stat_overlap_cycles),.stat_mac_count(stat_mac_count)
        );
    end else begin: raw_dut
        assign param_ready=1'b0;
        npu_gemv_top #(.LANES(LANES)) dut (
            .clk(clk),.rst(rst),.cmd_valid(cmd_valid),.cmd_ready(cmd_ready),
            .cmd_n(cmd_n),.cmd_k(cmd_k),
            .operand_valid(operand_valid),.operand_ready(operand_ready),
            .a_scalar(a_scalar),.b_vector(b_vector),
            .busy(busy),.done(done),.error(error),.out_valid(out_valid),
            .out_ready(out_ready),.out_row(out_row),.out_col(out_col),
            .out_data(out_data),.out_last(out_last),
            .stat_cycles(stat_cycles),.stat_load_cycles(stat_load_cycles),
            .stat_compute_cycles(stat_compute_cycles),
            .stat_input_wait_cycles(stat_input_wait_cycles),
            .stat_buffer_stall_cycles(stat_buffer_stall_cycles),
            .stat_output_stall_cycles(stat_output_stall_cycles),
            .stat_overlap_cycles(stat_overlap_cycles),.stat_mac_count(stat_mac_count)
        );
    end endgenerate

    reg signed [31:0] expected[0:65534];
    integer jobs=0;
    reg consecutive_mode=0;

    function automatic [31:0] mix(input reg [31:0] value);
        reg [31:0] x;
        begin
            x=value;
            x=x^(x<<13);x=x^(x>>17);x=x^(x<<5);
            mix=x;
        end
    endfunction
    function automatic integer av(input integer kk,seed,mode);
        reg [31:0] x;
        begin
            if(mode==4) av=0;
            else if(mode==1) av=1;
            else if(mode==2) av=-128;
            else if(mode==3) begin
                x=mix(32'h9187c432+kk*32'h873ba359+seed*32'habcde713);
                av=$signed(x[7:0]);
            end else av=((kk*13+seed*17)%256)-128;
        end
    endfunction
    function automatic integer bv(input integer kk,col,seed,mode);
        reg [31:0] x;
        begin
        if(mode==1) begin
            case(col%12)
                0,4,9: bv=1;
                1,5,10: bv=-1;
                2,6: bv=127;
                3,7: bv=-128;
                8: bv=0;
                default: bv=3;
            endcase
        end else if(mode==2) bv=-128;
        else if(mode==3) begin
            x=mix(32'h14a8e7cf+kk*32'h4fd79817+col*32'h371843b1+seed*32'h12387a51);
            bv=$signed(x[7:0]);
        end
        else bv=((kk*29+col*7+seed*23)%256)-128;
        end
    endfunction
    function automatic integer bias_for(input integer col,mode);
        if(mode==1) begin
            case(col%12)
                2,4,8: bias_for=32'sh7fffffff;
                3,5,9: bias_for=32'sh80000000;
                6: bias_for=12345;
                7: bias_for=-12345;
                10: bias_for=-1;
                default: bias_for=0;
            endcase
        end else if(mode==2) bias_for=32'sh7fffffff;
        else bias_for=((col*37)%511)-255;
    endfunction
    function automatic integer multiplier_for(input integer col,mode);
        if(mode==1 && (col%12==4 || col%12==5 || col%12==8 || col%12==9))
            multiplier_for=32'sh7fffffff;
        else if(mode==1 || mode==2) multiplier_for=1;
        else multiplier_for=(col%5)+1;
    endfunction
    function automatic integer shift_for(input integer col,mode);
        if(mode==1) begin
            case(col%12)
                0,1,11: shift_for=1;
                2,3,4,5,8,9: shift_for=31;
                default: shift_for=0;
            endcase
        end else if(mode==2) shift_for=31;
        else shift_for=(col%6)+3;
    endfunction
    // Division-based reference is independent of the DUT's magnitude/shift logic.
    function automatic integer quantize(input integer value,col,mode,relu_en);
        reg signed [63:0] sum,p,denom,q;
        integer s;
        begin
            sum=value;
            sum=sum+$signed(bias_for(col,mode));
            p=sum*$signed(multiplier_for(col,mode));
            s=shift_for(col,mode);
            denom=64'sd1<<s;
            if(p<0) q=-((-p+denom/2)/denom);
            else q=(p+denom/2)/denom;
            if(relu_en && q<0) q=0;
            if(q>127) quantize=127;
            else if(q< -128) quantize=-128;
            else quantize=q;
        end
    endfunction

    // stalls controls general input/parameter gaps and output backpressure.
    // Every job intentionally holds its final result for at least five cycles;
    // stalls=0 is therefore a directed handshake test, not a cycle benchmark.
    task automatic run_job(input integer nn,kk,relu_en,mode,seed,stalls,bad_params);
        integer c,k,l,groups,total_beats,sent,received,params,cycle;
        integer input_waits,output_stalls,last_holds,bad_phase;
        integer expected_value;
        reg hold_input,hold_param,in_fire,p_fire,bad_fire,out_fire,saw_done;
        reg stalled;
        reg signed [31:0] held_data;
        reg [15:0] held_row,held_col;
        reg held_last;
        reg [511:0] frozen_stats;
        begin
            for(c=0;c<nn;c=c+1) begin
                expected[c]=0;
                for(k=0;k<kk;k=k+1)
                    expected[c]=expected[c]+av(k,seed,mode)*bv(k,c,seed,mode);
            end
            @(negedge clk);
            if(!cmd_ready || busy || out_valid) $fatal(1,"job start not idle");
            cmd_n=nn;cmd_k=kk;cmd_relu=relu_en;cmd_valid=1;
            operand_valid=0;param_valid=0;out_ready=0;
            @(posedge clk);#1;
            if(!busy || cmd_ready || done || error) $fatal(1,"valid command not accepted");
            groups=(nn+LANES-1)/LANES;total_beats=groups*kk;
            sent=0;received=0;params=0;cycle=0;input_waits=0;output_stalls=0;
            last_holds=0;hold_input=0;hold_param=0;in_fire=0;p_fire=0;
            stalled=0;saw_done=0;bad_phase=bad_params?0:-1;
            while(!saw_done && cycle<total_beats+nn*12+20000) begin
                @(negedge clk);
                // A different command while busy must have no effect.
                cmd_valid=(cycle==3);
                if(cmd_valid) begin cmd_n=1;cmd_k=1;cmd_relu=!relu_en;end
                if(in_fire) hold_input=0;
                if(!hold_input && sent<total_beats &&
                   (!stalls || (cycle%7!=0 && cycle%7!=1))) hold_input=1;
                operand_valid=hold_input;
                a_scalar=av(sent%kk,seed,mode);
                for(l=0;l<LANES;l=l+1)
                    // Deliberately nonzero padded lanes exercise hardware masking.
                    b_vector[l*8+:8]=bv(sent%kk,(sent/kk)*LANES+l,seed,mode);
                if(p_fire) hold_param=0;
                if(!hold_param && params<nn &&
                   (!stalls || (cycle>11 && cycle%5!=0))) hold_param=1;
                param_valid=LINEAR && hold_param;
                param_bias=bias_for(params,mode);
                param_multiplier=multiplier_for(params,mode);
                param_shift=shift_for(params,mode);
                if(bad_phase==0) param_multiplier=0;
                if(bad_phase==1) param_multiplier=-1;
                if(bad_phase==2) param_shift=32;
                if(bad_phase==3) param_shift=63;
                if(out_valid && out_last && last_holds<5) out_ready=0;
                else out_ready=!stalls || (cycle%11!=0 && cycle%11!=1 && cycle%11!=2);
                @(posedge clk);
                if(stalled && (!out_valid || out_data!==held_data ||
                   out_row!==held_row || out_col!==held_col || out_last!==held_last))
                    $fatal(1,"output changed during backpressure");
                if(cmd_ready) $fatal(1,"command ready while busy");
                in_fire=operand_valid && operand_ready;
                p_fire=param_valid && param_ready;
                bad_fire=p_fire && bad_phase>=0;
                out_fire=out_valid && out_ready;
                if(operand_ready && !operand_valid) input_waits=input_waits+1;
                if(out_valid && !out_ready) output_stalls=output_stalls+1;
                if(out_valid && out_last && !out_ready) last_holds=last_holds+1;
                if(in_fire) sent=sent+1;
                if(p_fire) begin
                    if(bad_phase>=0) begin
                        if(bad_phase==3) bad_phase=-1;
                        else bad_phase=bad_phase+1;
                    end else params=params+1;
                end
                if(out_fire) begin
                    if(received>=nn) $fatal(1,"extra output");
                    expected_value=LINEAR?quantize(expected[received],received,mode,relu_en):expected[received];
                    if(out_row!==16'd0 || out_col!==received || out_data!==expected_value ||
                       out_last!==(received==nn-1))
                        $fatal(1,"result mismatch job=%0d col=%0d got=%0d expected=%0d tag=%0d,%0d last=%0b",
                               jobs,received,out_data,expected_value,out_row,out_col,out_last);
                    received=received+1;
                end
                stalled=out_valid && !out_ready;
                held_data=out_data;held_row=out_row;held_col=out_col;held_last=out_last;
                cycle=cycle+1;
                #1;
                if(error!==bad_fire) $fatal(1,"unexpected error state cycle=%0d bad_fire=%0b",cycle,bad_fire);
                if(done) begin
                    if(received!=nn || !out_fire || !held_last || busy || out_valid)
                        $fatal(1,"done preceded final result handshake");
                    saw_done=1;
                end else if(!busy) $fatal(1,"busy dropped before done");
            end
            if(!saw_done) $fatal(1,"job timed out N=%0d K=%0d sent=%0d received=%0d",nn,kk,sent,received);
            if(sent!=total_beats || (LINEAR && params!=nn) || last_holds<5)
                $fatal(1,"incomplete transfers or last result was not held");
            if(stat_cycles!==cycle || stat_load_cycles!==total_beats ||
               stat_compute_cycles!==total_beats || stat_overlap_cycles!==total_beats ||
               stat_mac_count!==(64'(nn)*kk) || stat_buffer_stall_cycles!==0 ||
               stat_input_wait_cycles!==input_waits || stat_output_stall_cycles!==output_stalls)
                $fatal(1,"counter mismatch cycles=%0d/%0d load=%0d compute=%0d overlap=%0d mac=%0d wait=%0d/%0d stall=%0d/%0d",
                       stat_cycles,cycle,stat_load_cycles,stat_compute_cycles,stat_overlap_cycles,
                       stat_mac_count,stat_input_wait_cycles,input_waits,stat_output_stall_cycles,output_stalls);
            if(stat_load_cycles>stat_cycles || stat_compute_cycles>stat_cycles ||
               stat_input_wait_cycles>stat_cycles || stat_output_stall_cycles>stat_cycles ||
               stat_overlap_cycles>stat_cycles) $fatal(1,"cycle counter exceeded job cycles");
            if(!consecutive_mode) begin
                frozen_stats={stat_cycles,stat_load_cycles,stat_compute_cycles,
                              stat_input_wait_cycles,stat_buffer_stall_cycles,
                              stat_output_stall_cycles,stat_overlap_cycles,stat_mac_count};
                @(negedge clk);cmd_valid=0;operand_valid=0;param_valid=0;out_ready=0;
                repeat(3) begin
                    @(posedge clk);#1;
                    if(done || error || busy || out_valid ||
                       {stat_cycles,stat_load_cycles,stat_compute_cycles,
                        stat_input_wait_cycles,stat_buffer_stall_cycles,
                        stat_output_stall_cycles,stat_overlap_cycles,stat_mac_count}!==frozen_stats)
                        $fatal(1,"idle state/counters did not freeze");
                end
            end
            $display("PASS GEMV LANES=%0d LINEAR=%0d N=%0d K=%0d mode=%0d relu=%0d cycles=%0d",
                     LANES,LINEAR,nn,kk,mode,relu_en,cycle);
            jobs=jobs+1;
        end
    endtask

    task automatic zero_command(input integer nn,kk);
        reg [63:0] saved;
        begin
            saved=stat_cycles;
            @(negedge clk);cmd_n=nn;cmd_k=kk;cmd_valid=1;
            @(posedge clk);#1;
            if(!error || busy || done || out_valid || !cmd_ready || stat_cycles!==saved)
                $fatal(1,"zero command did not produce isolated error");
            @(negedge clk);cmd_valid=0;
            @(posedge clk);#1;
            if(error || busy || done) $fatal(1,"zero command error was not one cycle");
        end
    endtask

    task automatic abort_job(input integer at_output);
        integer sent,limit;
        begin
            @(negedge clk);cmd_n=1;cmd_k=at_output?1:17;cmd_valid=1;
            operand_valid=0;param_valid=0;out_ready=0;
            @(posedge clk);#1;
            @(negedge clk);cmd_valid=0;operand_valid=1;a_scalar=-128;b_vector={LANES{8'h80}};
            param_valid=LINEAR;param_bias=0;param_multiplier=1;param_shift=0;
            sent=0;limit=0;
            while((at_output?!out_valid:sent<4) && limit<100) begin
                @(posedge clk);
                if(operand_valid && operand_ready) sent=sent+1;
                #1;
                @(negedge clk);
                if(sent>=(at_output?1:4)) operand_valid=0;
                limit=limit+1;
            end
            if(limit==100) $fatal(1,"abort setup timed out");
            rst=1;
            @(posedge clk);#1;
            if(busy || done || error || out_valid || cmd_ready || operand_ready ||
               stat_cycles!==0 || stat_mac_count!==0 || stat_output_stall_cycles!==0)
                $fatal(1,"reset did not abort cleanly");
            @(negedge clk);rst=0;cmd_valid=0;operand_valid=0;param_valid=0;out_ready=0;
            @(posedge clk);#1;
            if(!cmd_ready || busy || done || out_valid) $fatal(1,"reset recovery failed");
        end
    endtask

    initial begin
        repeat(3) @(posedge clk);
        @(negedge clk);rst=0;
        zero_command(0,1);
        zero_command(1,0);
        consecutive_mode=1;
        run_job(1,1,0,0,1,1,LINEAR);
        run_job(LANES+1,15,0,0,2,1,0);
        consecutive_mode=0;
        run_job(2*LANES-1,16,0,0,3,0,0);
        run_job(2*LANES+1,17,0,0,4,1,0);
        run_job(LANES+1,17,0,4,37,1,0);
        run_job(3*LANES+2,23,0,3,83,1,0);
        run_job(2*LANES+3,9,1,3,149,1,0);
        run_job(13,1,0,1,0,1,LINEAR);
        run_job(13,1,1,1,0,1,0);
        abort_job(0);
        run_job(LANES+1,1,0,0,5,0,0);
        abort_job(1);
        run_job(1,65535,0,2,0,0,0);
        run_job(65535,1,0,0,6,0,0);
        zero_command(0,0);
        run_job(1,17,0,0,7,1,0);
        $display("PASS tb_npu_gemv LANES=%0d LINEAR=%0d jobs=%0d",LANES,LINEAR,jobs);
        $finish;
    end
    initial begin
        #100000000;
        $fatal(1,"global timeout");
    end
endmodule
`default_nettype wire
