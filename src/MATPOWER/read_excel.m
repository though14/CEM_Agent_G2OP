%\ T = readtable("C:\Users\thoug\OneDrive\SS2023\Internship\03_MATPOWER\01_File\Load_P.xlsx")
% 
% TT = readtable("C:\Users\thoug\OneDrive\SS2023\Internship\02_G2OP\File\chronics\0000\_N_loads_p.csv.bz2.csv")
% 


% when you want to turn the table into double array
% -->  TT_double=TT{:,:}

% Transpose the double array
% -->  TT_d_t = transpose(TT_double)


%%
%clear
clc

folder_path = "C:\Users\thoug\OneDrive\SS2023\Internship\04_Code\CEM_Agent_G2OP\File\MATPOWER\Chronic_rte_case14" ;

folder_path = "C:\Users\thoug\OneDrive\SS2023\Internship\04_Code\CEM_Agent_G2OP\File\MATPOWER\Chronic_rte_case14" ;
folder_name = 197;

num_folder= 10;
%%
% filePattern = fullfile(folder_path, compose("%04d", folder_name), '*__mod.csv');
% % filePattern = fullfile(folder_path,'**/*__mod.csv');
% 
% theFiles = dir(filePattern);

for i=000:1000 %it should be num_folder

    filePattern = fullfile(folder_path, compose("%03d", i), '*__mod.csv');

    theFiles = dir(filePattern);

    for k=1:length(theFiles)
        baseFileName = theFiles(k).name;
        fullFileName = fullfile(theFiles(k).folder, baseFileName);
        fprintf(1, 'Now reading %s\n', fullFileName);

        matFileName = strcat(fullFileName, '.mat');
    
        TT = readtable(fullFileName,'PreserveVariableNames',true);
        T_d = TT{:,:};
        T_d_t = transpose(T_d);

        save(matFileName, "T_d_t", '-mat')
        
    end

end

%%

%to load the .mat file

clc

for i=0000:0000 %it should be num_folder

    filePattern = fullfile(folder_path, compose("%03d", i), '*__mod.csv.mat');

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













