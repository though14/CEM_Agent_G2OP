%STARTUP

%% add MATPOWER paths
addpath( ...
    'C:\Users\thoug\OneDrive\SS2023\Internship\03_MATPOWER\lib', ...
    'C:\Users\thoug\OneDrive\SS2023\Internship\03_MATPOWER\lib\t', ...
    'C:\Users\thoug\OneDrive\SS2023\Internship\03_MATPOWER\data', ...
    'C:\Users\thoug\OneDrive\SS2023\Internship\03_MATPOWER\most\lib', ...
    'C:\Users\thoug\OneDrive\SS2023\Internship\03_MATPOWER\most\lib\t', ...
    'C:\Users\thoug\OneDrive\SS2023\Internship\03_MATPOWER\mp-opt-model\lib', ...
    'C:\Users\thoug\OneDrive\SS2023\Internship\03_MATPOWER\mp-opt-model\lib\t', ...
    'C:\Users\thoug\OneDrive\SS2023\Internship\03_MATPOWER\mips\lib', ...
    'C:\Users\thoug\OneDrive\SS2023\Internship\03_MATPOWER\mips\lib\t', ...
    'C:\Users\thoug\OneDrive\SS2023\Internship\03_MATPOWER\mptest\lib', ...
    'C:\Users\thoug\OneDrive\SS2023\Internship\03_MATPOWER\mptest\lib\t', ...
    'C:\Users\thoug\OneDrive\SS2023\Internship\03_MATPOWER\extras\maxloadlim', ...
    'C:\Users\thoug\OneDrive\SS2023\Internship\03_MATPOWER\extras\maxloadlim\tests', ...
    'C:\Users\thoug\OneDrive\SS2023\Internship\03_MATPOWER\extras\maxloadlim\examples', ...
    'C:\Users\thoug\OneDrive\SS2023\Internship\03_MATPOWER\extras\misc', ...
    'C:\Users\thoug\OneDrive\SS2023\Internship\03_MATPOWER\extras\reduction', ...
    'C:\Users\thoug\OneDrive\SS2023\Internship\03_MATPOWER\extras\sdp_pf', ...
    'C:\Users\thoug\OneDrive\SS2023\Internship\03_MATPOWER\extras\se', ...
    'C:\Users\thoug\OneDrive\SS2023\Internship\03_MATPOWER\extras\smartmarket', ...
    'C:\Users\thoug\OneDrive\SS2023\Internship\03_MATPOWER\extras\state_estimator', ...
    'C:\Users\thoug\OneDrive\SS2023\Internship\03_MATPOWER\extras\syngrid\lib', ...
    'C:\Users\thoug\OneDrive\SS2023\Internship\03_MATPOWER\extras\syngrid\lib\t', ...
    '-end' );


%%
%clear
clc

%YOU MUST CHANGE FOLDER PATH UPTO ...\0000  --> IN THIS CASE PATH GOES 
%TILL \chronics 
folder_path = "C:\Users\thoug\OneDrive\SS2023\Internship\02_G2OP\File\chronics" ;
folder_name = 0000;

num_folder= 1004;

%to load the .mat file

clc

for i=0000:0000 %it should be num_folder

    filePattern = fullfile(folder_path, compose("%04d", i), '*__mod.csv.mat');

    theFiles = dir(filePattern);

    for k=1:length(theFiles)
        baseFileName = theFiles(k).name;
        fullFileName = fullfile(theFiles(k).folder, baseFileName);
        fprintf(1, 'Now reading %s\n', fullFileName);

%         matFileName = strcat(fullFileName, '.mat');
%     
%         TT = readtable(fullFileName,'PreserveVariableNames',true);
%         T_d = TT{:,:};
%         T_d_t = transpose(T_d);
% 
%         save(matFileName, "T_d_t", '-mat')
        
        if fullFileName == fullfile(folder_path, compose("%04d",i), '_N_loads_p_planned.csv.bz2.csv__mod.csv.mat' )
            load_p_planned = struct2cell(load(fullFileName));
            load_p_planned = cat(2,load_p_planned{:});

    
        elseif fullFileName == fullfile(folder_path, compose("%04d",i),'_N_loads_q.csv.bz2.csv__mod.csv.mat')
            load_q = struct2cell(load(fullFileName));
            load_q = cat(2,load_q{:});


        elseif fullFileName == fullfile(folder_path, compose("%04d",i),'_N_loads_q_planned.csv.bz2.csv__mod.csv.mat')
            load_q_planned = struct2cell(load(fullFileName));
            load_q_planned = cat(2,load_q_planned{:});

        elseif fullFileName == fullfile(folder_path, compose("%04d",i),'_N_loads_p.csv.bz2.csv__mod.csv.mat')
            load_p = struct2cell(load(fullFileName));
            load_p = cat(2, load_p{:});

        elseif fullFileName == fullfile(folder_path, compose("%04d",i),'_N_prods_p.csv.bz2.csv__mod.csv.mat')
            gen_p = struct2cell(load(fullFileName));
            gen_p = cat(2,gen_p{:});
        elseif fullFileName == fullfile(folder_path, compose("%04d",i),'_N_prods_p_planned.csv.bz2.csv__mod.csv.mat')
            gen_p_planned = struct2cell(load(fullFileName));
            gen_p_planned = cat(2,gen_p_planned{:});
        elseif fullFileName == fullfile(folder_path, compose("%04d",i),'_N_prods_v.csv.bz2.csv__mod.csv.mat')
            gen_v = struct2cell(load(fullFileName));
            gen_v = cat(2,gen_v{:});
            gen_v = gen_v/100;
        elseif fullFileName == fullfile(folder_path, compose("%04d",i),'_N_prods_v_planned.csv.bz2.csv__mod.csv.mat')
            gen_v_planned = struct2cell(load(fullFileName));
            gen_v_planned = cat(2,gen_v_planned{:});
        elseif fullFileName == fullfile(folder_path, compose("%04d",i),'_N_simu_ids.csv.bz2.csv__mod.csv.mat')
            simu_id = struct2cell(load(fullFileName));
            simu_id = cat(2,simu_id{:});
        elseif fullFileName == fullfile(folder_path, compose("%04d",i),'hazards.csv.bz2.csv__mod.csv')
            harzard = struct2cell(load(fullFileName));
            harzard = cat(2,harzard{:});
        elseif fullFileName == fullfile(folder_path, compose("%04d",i),'maintenance.csv.bz2.csv__mod.csv.mat')
            maintenance = struct2cell(load(fullFileName));
            maintenance = cat(2,maintenance{:});
        elseif fullFileName == fullfile(folder_path, compose("%04d",i), '_N_imaps.csv.bz2.csv__mod.csv.mat' )
            line = struct2cell(load(fullFileName));
            line = cat(2,line{:});
        

        end
            
        
        

        
        
    end

end

%%
%%

%load mpc file(grid file)

%MUST CHANGE THE PATH FOR THE GRID FILEZ

mpc = struct2cell(load("C:\Users\thoug\OneDrive\SS2023\Internship\03_MATPOWER\01_File\rte_case14_thermal_update.mat"));
mpc = cat(2,mpc{:});

v_base = 100
thermal_limit = [1000,1000,1000,1000,1000,1000,1000,760,380,380,760,450,760,2000,2000,380,380,760,760,380]

updated_thermal_limit = (thermal_limit * v_base)/1000

for k = 6:8
    mpc.branch(:,k)= updated_thermal_limit

end


%%
%try update values in mpc

number_TS = 500;

for i = 1:number_TS
    
    %Bus
    mpc.bus(:,3) = load_p(:,i);
    mpc.bus(:,4) = load_q(:,i);



    %Gen
    mpc.gen(1,2) = gen_p(2,i);
    mpc.gen(2,2) = gen_p(3,i);
    mpc.gen(3,2) = gen_p(6,i);
    mpc.gen(4,2) = gen_p(8,i);
    mpc.gen(5,2) = gen_p(1,i);

    mpc.gen(1,6) = gen_v(2,i);
    mpc.gen(2,6) = gen_v(3,i);
    mpc.gen(3,6) = gen_v(6,i);
    mpc.gen(4,6) = gen_v(8,i);
    mpc.gen(5,6) = gen_v(1,i);


    result_pf(i) = runpf(mpc);

end
