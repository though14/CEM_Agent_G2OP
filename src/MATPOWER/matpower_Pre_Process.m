
%make array of sub_minus that doesn't include sub_info
All_row = (1:211).';
Linear_indices= sub;
idx=ismember(1:numel(All_row),Linear_indices); % idx is logical indices
All_row(idx) = [];

sub_minus = All_row;

% Don't use this, use from row 14
% BB = [33,52,65,84,100,119,132,142,158,168,178,188,201,211]  %from original BR_STATUS
% BB_1 = [5,12,17,24,30,37,42,46,52,56,60,64,69,73] %if we work from sub_minus that already take out sth

All_row_bb = (1:211).';
sub_wo_bb = [25,26,27,28,40,41,42,43,44,45,57,58,59,60,72,73,74,75,76,77,90,91,92,93,94,107,108,109,110,111,112,124,125,126,127,136,137,138,148,149,150,151,152,162,163,164,172,173,174,182,183,184,193,194,195,196,205,206,207 ];
%substation without bus-bus breaker
idx_bb=ismember(1:numel(All_row_bb),sub_wo_bb);
All_row_bb(idx_bb) = [];

sub_wo_bb_info = All_row_bb;

All_row_bb_load = (1:211).';
sub_wo_bb_right_load = [25,26,27,40,41,42,43,44,45,57,58,59,60,72,73,74,75,76,77,90,91,92,93,94,107,108,109,110,111,112,124,125,126,136,137,148,149,150,151,152,162,163,164,172,173,174,182,183,184,193,194,195,196,205,206,207 ];
idx_bb_load = ismember(1:numel(All_row_bb_load),sub_wo_bb_right_load);
All_row_bb_load(idx_bb_load) = [];

sub_wo_bb_load = All_row_bb_load;



%%
for i = sub_wo_bb_load  %we can put here whatever we want from sub_wo_bb_info,sub_wo_bb_load
    
    BR_STATUS_Work = BR_STATUS;
    BR_STATUS_Work(i,:) = [];

end

%in case to check with manual worked one : isequal(BR_STATUS_Work,
%BR_STATUS_Relavant_wo_BBDC) 
%if answer is 1, it is equal