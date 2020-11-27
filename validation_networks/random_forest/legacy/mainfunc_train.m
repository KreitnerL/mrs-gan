%%
% Main function to train and generate training models
% The arguments include the reduced and oblique dictionaries and the
% training labels
% The output includes the training models. These are passed on subsequently as arguments
% to the functions for different datasets in which the testing is performed
% 
% refer to regRF_spec_v4 for full script (original)
% (c) Dhritiman Das
% Munich, 2018
%%
X = dictionaryReduced; % X should be NxL, where N = number of observations, L = spectral length
%X_obl = dict_obl;
%noise_val = [0.01 0.05 0.15 0.25];

sz_sp = size(X,2);

dat_ind = randperm(1000); %Originally 14400

%%
% For Full Spectrum
X_test = X(dat_ind, :);
X_tr = X;
X_tr(dat_ind, :) = [];

X_test_full = dictionary(dat_ind, :);
X_tr_full = dictionary;
X_tr_full(dat_ind,:) = [];

%%
% relative conc.
Y_pch = rlabels(:, 1);
Y_pch_ts = Y_pch(dat_ind);
Y_pch_tr = Y_pch;
Y_pch_tr(dat_ind) = [];


Y_naa  = rlabels(:, 3);
Y_naa_ts = Y_naa(dat_ind);
Y_naa_tr = Y_naa;
Y_naa_tr(dat_ind) = [];


Y_glx = rlabels(:, 7);
Y_glx_ts = Y_glx(dat_ind);
Y_glx_tr = Y_glx;
Y_glx_tr(dat_ind) = [];

Y_ins = rlabels(:, 8);
Y_ins_ts = Y_ins(dat_ind);
Y_ins_tr = Y_ins;
Y_ins_tr(dat_ind) = [];

Y_snr = rlabels(:, 5);
Y_snr_ts = Y_snr(dat_ind);
Y_snr_tr = Y_snr;
Y_snr_tr(dat_ind) = [];

Y_t2 = rlabels(:, 6);
Y_t2_ts = Y_t2(dat_ind);
Y_t2_tr = Y_t2;
Y_t2_tr(dat_ind) = [];

% Y_ace = rlabels(:, 10);
% Y_ace_ts = Y_ace(dat_ind);
% Y_ace_tr = Y_ace;
% Y_ace_tr(dat_ind) = [];

% Y_ala = rlabels(:, 11);
% Y_ala_ts = Y_ala(dat_ind);
% Y_ala_tr = Y_ala;
% Y_ala_tr(dat_ind) = [];
% 
% Y_asc = rlabels(:, 12);
% Y_asc_ts = Y_asc(dat_ind);
% Y_asc_tr = Y_asc;
% Y_asc_tr(dat_ind) = [];
% 
% Y_asp = rlabels(:, 13);
% Y_asp_ts = Y_asp(dat_ind);
% Y_asp_tr = Y_asp;
% Y_asp_tr(dat_ind) = [];
% 
% Y_gaba = rlabels(:, 14);
% Y_gaba_ts = Y_gaba(dat_ind);
% Y_gaba_tr = Y_gaba;
% Y_gaba_tr(dat_ind) = [];
% 
% Y_glc = rlabels(:, 15);
% Y_glc_ts = Y_glc(dat_ind);
% Y_glc_tr = Y_glc;
% Y_glc_tr(dat_ind) = [];
% 
% Y_gln = rlabels(:, 16);
% Y_gln_ts = Y_gln(dat_ind);
% Y_gln_tr = Y_gln;
% Y_gln_tr(dat_ind) = [];
% 
% Y_glu = rlabels(:, 17);
% Y_glu_ts = Y_glu(dat_ind);
% Y_glu_tr = Y_glu;
% Y_glu_tr(dat_ind) = [];
% 
% Y_gly = rlabels(:, 18);
% Y_gly_ts = Y_gly(dat_ind);
% Y_gly_tr = Y_gly;
% Y_gly_tr(dat_ind) = [];
% 
% Y_gpc = rlabels(:, 19);
% Y_gpc_ts = Y_gpc(dat_ind);
% Y_gpc_tr = Y_gpc;
% Y_gpc_tr(dat_ind) = [];
% 
% Y_gsh = rlabels(:, 20);
% Y_gsh_ts = Y_gsh(dat_ind);
% Y_gsh_tr = Y_gsh;
% Y_gsh_tr(dat_ind) = [];
% 
% Y_lac = rlabels(:, 21);
% Y_lac_ts = Y_lac(dat_ind);
% Y_lac_tr = Y_lac;
% Y_lac_tr(dat_ind) = [];
% 
% Y_naag = rlabels(:, 22);
% Y_naag_ts = Y_naag(dat_ind);
% Y_naag_tr = Y_naag;
% Y_naag_tr(dat_ind) = [];
% 
% Y_pcr = rlabels(:, 23);
% Y_pcr_ts = Y_pcr(dat_ind);
% Y_pcr_tr = Y_pcr;
% Y_pcr_tr(dat_ind) = [];
% 
% Y_pe = rlabels(:, 24);
% Y_pe_ts = Y_pe(dat_ind);
% Y_pe_tr = Y_pe;
% Y_pe_tr(dat_ind) = [];
% 
% Y_sIns = rlabels(:, 25);
% Y_sIns_ts = Y_sIns(dat_ind);
% Y_sIns_tr = Y_sIns;
% Y_sIns_tr(dat_ind) = [];
% 
% Y_tau = rlabels(:, 26);
% Y_tau_ts = Y_tau(dat_ind);
% Y_tau_tr = Y_tau;
% Y_tau_tr(dat_ind) = [];

%%
tr_label_rel.naa = Y_naa_tr;
tr_label_rel.pch = Y_pch_tr;
tr_label_rel.glx = Y_glx_tr;
tr_label_rel.ins = Y_ins_tr;
tr_label_rel.snr = Y_snr_tr;
tr_label_rel.t2 = Y_t2_tr;
% tr_label_rel.ace = Y_ace_tr;
% tr_label_rel.ala = Y_ala_tr;
% tr_label_rel.asc = Y_asc_tr;
% tr_label_rel.asp = Y_asp_tr;
% tr_label_rel.gaba = Y_gaba_tr;
% tr_label_rel.glc = Y_glc_tr;
% tr_label_rel.gln = Y_gln_tr;
% tr_label_rel.glu = Y_glu_tr;
% tr_label_rel.gly = Y_gly_tr;
% tr_label_rel.gpc = Y_gpc_tr;
% tr_label_rel.gsh = Y_gsh_tr;
% tr_label_rel.lac = Y_lac_tr;
% tr_label_rel.naag = Y_naag_tr;
% tr_label_rel.pcr = Y_pcr_tr;
% tr_label_rel.pe = Y_pe_tr;
% tr_label_rel.sIns = Y_sIns_tr;
% tr_label_rel.tau = Y_tau_tr;


ts_label_rel.naa = Y_naa_ts;
ts_label_rel.pch = Y_pch_ts;
ts_label_rel.glx = Y_glx_ts;
ts_label_rel.ins = Y_ins_ts;
ts_label_rel.snr = Y_snr_ts;
ts_label_rel.t2 = Y_t2_ts;
% ts_label_rel.ace = Y_ace_ts;
% ts_label_rel.ala = Y_ala_ts;
% ts_label_rel.asc = Y_asc_ts;
% ts_label_rel.asp = Y_asp_ts;
% ts_label_rel.gaba = Y_gaba_ts;
% ts_label_rel.glc = Y_glc_ts;
% ts_label_rel.gln = Y_gln_ts;
% ts_label_rel.glu = Y_glu_ts;
% ts_label_rel.gly = Y_gly_ts;
% ts_label_rel.gpc = Y_gpc_ts;
% ts_label_rel.gsh = Y_gsh_ts;
% ts_label_rel.lac = Y_lac_ts;
% ts_label_rel.naag = Y_naag_ts;
% ts_label_rel.pcr = Y_pcr_ts;
% ts_label_rel.pe = Y_pe_ts;
% ts_label_rel.sIns = Y_sIns_ts;
% ts_label_rel.tau = Y_tau_ts;

%clearvars Y_pch Y_cr Y_naa Y_glx Y_ins Y_snr Y_t2 Y_ace Y_ala Y_asc Y_asp Y_gaba Y_glc Y_gln Y_glu Y_gly Y_gpc Y_gsh Y_lac Y_naag Y_pcr Y_pe Y_sIns Y_tau

%%
%---- Abs conc--------%

% Y_pch = labels(:, 1);
% Y_pch_ts = Y_pch(dat_ind);
% Y_pch_tr = Y_pch;
% Y_pch_tr(dat_ind) = [];
% 
% 
% Y_naa  = labels(:, 3);
% Y_naa_ts = Y_naa(dat_ind);
% Y_naa_tr = Y_naa;
% Y_naa_tr(dat_ind) = [];
% 
% 
% Y_glx = labels(:, 7);
% Y_glx_ts = Y_glx(dat_ind);
% Y_glx_tr = Y_glx;
% Y_glx_tr(dat_ind) = [];
% 
% Y_ins = labels(:, 8);
% Y_ins_ts = Y_ins(dat_ind);
% Y_ins_tr = Y_ins;
% Y_ins_tr(dat_ind) = [];
% 
% Y_snr = labels(:, 5);
% Y_snr_ts = Y_snr(dat_ind);
% Y_snr_tr = Y_snr;
% Y_snr_tr(dat_ind) = [];
% 
% Y_t2 = labels(:, 6);
% Y_t2_ts = Y_t2(dat_ind);
% Y_t2_tr = Y_t2;
% Y_t2_tr(dat_ind) = [];
% 
% Y_ace = labels(:, 10);
% Y_ace_ts = Y_ace(dat_ind);
% Y_ace_tr = Y_ace;
% Y_ace_tr(dat_ind) = [];
% 
% Y_ala = labels(:, 11);
% Y_ala_ts = Y_ala(dat_ind);
% Y_ala_tr = Y_ala;
% Y_ala_tr(dat_ind) = [];
% 
% Y_asc = labels(:, 12);
% Y_asc_ts = Y_asc(dat_ind);
% Y_asc_tr = Y_asc;
% Y_asc_tr(dat_ind) = [];
% 
% Y_asp = labels(:, 13);
% Y_asp_ts = Y_asp(dat_ind);
% Y_asp_tr = Y_asp;
% Y_asp_tr(dat_ind) = [];
% 
% Y_gaba = labels(:, 14);
% Y_gaba_ts = Y_gaba(dat_ind);
% Y_gaba_tr = Y_gaba;
% Y_gaba_tr(dat_ind) = [];
% 
% Y_glc = labels(:, 15);
% Y_glc_ts = Y_glc(dat_ind);
% Y_glc_tr = Y_glc;
% Y_glc_tr(dat_ind) = [];
% 
% Y_gln = labels(:, 16);
% Y_gln_ts = Y_gln(dat_ind);
% Y_gln_tr = Y_gln;
% Y_gln_tr(dat_ind) = [];
% 
% Y_glu = labels(:, 17);
% Y_glu_ts = Y_glu(dat_ind);
% Y_glu_tr = Y_glu;
% Y_glu_tr(dat_ind) = [];
% 
% Y_gly = labels(:, 18);
% Y_gly_ts = Y_gly(dat_ind);
% Y_gly_tr = Y_gly;
% Y_gly_tr(dat_ind) = [];
% 
% Y_gpc = labels(:, 19);
% Y_gpc_ts = Y_gpc(dat_ind);
% Y_gpc_tr = Y_gpc;
% Y_gpc_tr(dat_ind) = [];
% 
% Y_gsh = labels(:, 20);
% Y_gsh_ts = Y_gsh(dat_ind);
% Y_gsh_tr = Y_gsh;
% Y_gsh_tr(dat_ind) = [];
% 
% Y_lac = labels(:, 21);
% Y_lac_ts = Y_lac(dat_ind);
% Y_lac_tr = Y_lac;
% Y_lac_tr(dat_ind) = [];
% 
% 
% Y_naag = labels(:, 22);
% Y_naag_ts = Y_naag(dat_ind);
% Y_naag_tr = Y_naag;
% Y_naag_tr(dat_ind) = [];
% 
% Y_pcr = labels(:, 23);
% Y_pcr_ts = Y_pcr(dat_ind);
% Y_pcr_tr = Y_pcr;
% Y_pcr_tr(dat_ind) = [];
% 
% Y_pe = labels(:, 24);
% Y_pe_ts = Y_pe(dat_ind);
% Y_pe_tr = Y_pe;
% Y_pe_tr(dat_ind) = [];
% 
% Y_sIns = labels(:, 25);
% Y_sIns_ts = Y_sIns(dat_ind);
% Y_sIns_tr = Y_sIns;
% Y_sIns_tr(dat_ind) = [];
% 
% Y_tau = labels(:, 26);
% Y_tau_ts = Y_tau(dat_ind);
% Y_tau_tr = Y_tau;
% Y_tau_tr(dat_ind) = [];
% %%
% tr_label_abs.naa = Y_naa_tr;
% tr_label_abs.pch = Y_pch_tr;
% tr_label_abs.glx = Y_glx_tr;
% tr_label_abs.ins = Y_ins_tr;
% tr_label_abs.snr = Y_snr_tr;
% tr_label_abs.t2 = Y_t2_tr;
% tr_label_abs.ace = Y_ace_tr;
% tr_label_abs.ala = Y_ala_tr;
% tr_label_abs.asc = Y_asc_tr;
% tr_label_abs.asp = Y_asp_tr;
% tr_label_abs.gaba = Y_gaba_tr;
% tr_label_abs.glc = Y_glc_tr;
% tr_label_abs.gln = Y_gln_tr;
% tr_label_abs.glu = Y_glu_tr;
% tr_label_abs.gly = Y_gly_tr;
% tr_label_abs.gpc = Y_gpc_tr;
% tr_label_abs.gsh = Y_gsh_tr;
% tr_label_abs.lac = Y_lac_tr;
% tr_label_abs.naag = Y_naag_tr;
% tr_label_abs.pcr = Y_pcr_tr;
% tr_label_abs.pe = Y_pe_tr;
% tr_label_abs.sIns = Y_sIns_tr;
% tr_label_abs.tau = Y_tau_tr;
% 
% 
% ts_label_abs.naa = Y_naa_ts;
% ts_label_abs.pch = Y_pch_ts;
% ts_label_abs.glx = Y_glx_ts;
% ts_label_abs.ins = Y_ins_ts;
% ts_label_abs.snr = Y_snr_ts;
% ts_label_abs.t2 = Y_t2_ts;
% ts_label_abs.ace = Y_ace_ts;
% ts_label_abs.ala = Y_ala_ts;
% ts_label_abs.asc = Y_asc_ts;
% ts_label_abs.asp = Y_asp_ts;
% ts_label_abs.gaba = Y_gaba_ts;
% ts_label_abs.glc = Y_glc_ts;
% ts_label_abs.gln = Y_gln_ts;
% ts_label_abs.glu = Y_glu_ts;
% ts_label_abs.gly = Y_gly_ts;
% ts_label_abs.gpc = Y_gpc_ts;
% ts_label_abs.gsh = Y_gsh_ts;
% ts_label_abs.lac = Y_lac_ts;
% ts_label_abs.naag = Y_naag_ts;
% ts_label_abs.pcr = Y_pcr_ts;
% ts_label_abs.pe = Y_pe_ts;
% ts_label_abs.sIns = Y_sIns_ts;
% ts_label_abs.tau = Y_tau_ts;

%%
% Clear variables which aren't needed
clearvars Y_pch Y_cr Y_naa Y_glx Y_ins
%clearvars dictionary dictionaryReduced dict_obl 
%% 
% Take absolute or real values
%note = {'50k spec: SynSpec_bestSNR_v18_250518.mat'};
X_tr = (X_tr(:,:)); %originally real
X_test = (X_test(:,:));

%
%X_tr_obl = real(X_tr_obl(:,:));
%X_test_obl = real(X_test_obl(:,:));

%%
% Preparing training data
train_samp = randperm(size(X_tr,1));

%% Experiment to plot OOB vs Mtry for full data

% mtry = [32,64,87,128];
% mtry_exp_time = 0;
% 
% 
% 
%  X_mt_trn = X_tr(train_samp(1:20000),:);
%  Y_mt_trn_naa = Y_naa_tr(train_samp(1:20000),:);
%  Y_mt_trn_pch = Y_pch_tr(train_samp(1:20000),:);
% % Y_mt_trn_cr = Y_cr_tr(train_samp(1:40000),:);
%  Y_mt_trn_glx = Y_glx_tr(train_samp(1:20000),:);
%   Y_mt_trn_ins = Y_ins_tr(train_samp(1:20000),:);
% 
%  
%  
%  % Running loop for generating RF Reg models for different mTry(features)
% for num = 1:length(mtry)
%     tic;
%     
%      model_mtr_naa{num} =  regRF_train(double(X_mt_trn), double(Y_mt_trn_naa), 201, mtry(num));
%      fprintf('\n count naa: %f\n', num)
%      model_mtr_pch{num} =  regRF_train(double(X_mt_trn), double(Y_mt_trn_pch), 201, mtry(num));
%      fprintf('\n count pch: %f\n', num)
%     % model_mtr_cr{num} =  regRF_train(double(X_mt_trn), double(Y_mt_trn_cr), 201, mtry(num));
%     % fprintf('\n count cr: %f\n', num)
%      
%      model_mtr_glx{num} =  regRF_train(double(X_mt_trn), double(Y_mt_trn_glx), 201, mtry(num));
%      fprintf('\n count glx: %f\n', num)
%      
%      model_mtr_ins{num} =  regRF_train(double(X_mt_trn), double(Y_mt_trn_ins), 201, mtry(num));
%      fprintf('\n count inositol: %f\n', num)
%      
%      
%     % save('models_oob_exp_230517.mat','model_mtr_naa','model_mtr_pch','model_mtr_cr', 'model_mtr_t2', 'model_mtr_glx', 'model_mtr_ins')
%     % fprintf('\n saved iteration: %f\n', num)
%      
%      mtry_exp_time(num) = toc;
%      
%      oob_mod_naa{num} = model_mtr_naa{1,num}.mse;
%      figure(8), plot(oob_mod_naa{num}), drawnow, hold on
% 
%      oob_mod_pch{num} = model_mtr_pch{1,num}.mse;
%      figure(9), plot(oob_mod_pch{num}), drawnow, hold on
%      
%     % oob_mod_cr{num} = model_mtr_cr{1,num}.mse;
%     % figure(10), plot(oob_mod_cr{num}), drawnow, hold on
%      
%      oob_mod_glx{num} = model_mtr_glx{1,num}.mse;
%      figure(11), plot(oob_mod_glx{num}), drawnow, hold on
%      
%      oob_mod_ins{num} = model_mtr_ins{1,num}.mse;
%      figure(12), plot(oob_mod_ins{num}), drawnow, hold on
%      
%      
% end
% figure(8), xlabel('Number of trees'), ylabel('MSE- OOB'), title('NAA: OOB vs mtry for nTree=201'), legend('32','64','87','128')
% figure(9), xlabel('Number of trees'), ylabel('MSE- OOB'), title('pch: OOB vs mtry for nTree=201'), legend('32','64','87','128')
% %figure(10), xlabel('Number of trees'), ylabel('MSE- OOB'), title('Cr: OOB vs mtry for nTree=201'), legend('32','64','87','128')
% figure(11), xlabel('Number of trees'), ylabel('MSE- OOB'), title('Glx: OOB vs mtry for nTree=201'), legend('32','64','87','128')
% figure(12), xlabel('Number of trees'), ylabel('MSE- OOB'), title('Ins: OOB vs mtry for nTree=201'), legend('32','64','87','128')
% hold off
% 
% clearvars oob_mod_naa oob_mod_pch oob_mod_cr oob_mod_glx oob_mod_ins oob_mod_t2 oob_mod_mac
% 

%%

% Define data size and number of regression tress

data_sz = size(train_samp, 2); % compute total size of dictionary
samp_sz = 0.05;
samp_tot = data_sz * samp_sz;
reg_num = data_sz/samp_tot; %total number of regressors
nTrees = 100;

a = 1;
b = samp_tot;

%% 
%%UNIVARIATE METHOD
for i = 1:reg_num
  
    X_train{i} = X_tr(train_samp(a:b), :);
   
   
    Y_train_naa{i} = Y_naa_tr(train_samp(a:b), :);
    Y_train_pch{i} = Y_pch_tr(train_samp(a:b), :);
   % Y_train_t2{i} = Y_t2_tr(train_samp(a:b), :);
    Y_train_glx{i} = Y_glx_tr(train_samp(a:b), :);
    Y_train_ins{i} = Y_ins_tr(train_samp(a:b), :);
    %Y_train_cr{i} = Y_cr_tr(train_samp(a:b), :);
  %  Y_train_mac{i} = Y_mac_tr(train_samp(a:b), :);
     if b<(data_sz+1)
    a = a + samp_tot;
    b = a + samp_tot - 1;
    end
end   
   
%% RUN UNIVARIATE REGRESSION

for i = 1:reg_num
   model_naa{i} =  regRF_train(double(X_train{i}), double(Y_train_naa{i}), nTrees, 87); 
   fprintf('\n UV Naa count: %f\n', i)
   
   model_pch{i} =  regRF_train(double(X_train{i}), double(Y_train_pch{i}), nTrees, 87); 
   fprintf('\n UV PCh count: %f\n', i)
   
   model_glx{i} =  regRF_train(double(X_train{i}), double(Y_train_glx{i}), nTrees, 87); 
   fprintf('\n UV Glx count: %f\n', i)
   
   model_ins{i} =  regRF_train(double(X_train{i}), double(Y_train_ins{i}), nTrees, 87); 
   fprintf('\n UV Ins count: %f\n', i)
   
%   model_t2{i} =  regRF_train(double(X_train{i}), double(Y_train_t2{i}), nTrees, 87); 
   fprintf('\n UV T2 count: %f\n', i)
   
 %  model_mac{i} =  regRF_train(double(X_train{i}), double(Y_train_mac{i}), nTrees, 87); 
   fprintf('\n UV Mac count: %f\n', i)
   
    Est_naa{i} = regRF_predict(double(X_test), model_naa{i});
    Est_pch{i} = regRF_predict(double(X_test), model_pch{i});
 %   Est_t2{i} = regRF_predict(double(X_test), model_t2{i});
    Est_glx{i} = regRF_predict(double(X_test), model_glx{i});
    Est_ins{i} = regRF_predict(double(X_test), model_ins{i});
 %   Est_mac{i} = regRF_predict(double(X_test), model_mac{i});
     
    avg_est_naa(:,i) = Est_naa{i};;
    avg_est_pch(:,i) = Est_pch{i};
    avg_est_glx(:,i) = Est_glx{i};
    avg_est_ins(:,i) = Est_ins{i};
  %  avg_est_t2(:,i) = Est_t2{i};
  %  avg_est_mac(:,i) = Est_mac{i};
end
%%
save('Training_sim_27032018.mat', 'X_tr','X_test','dat_ind','train_samp', 'model_glx','model_ins','model_naa','model_pch','Y_naa_ts','Y_pch_ts','Y_glx_ts','Y_ins_ts')
%%clearvars model_naa model_pch model_t2 model_glx model_ins model_mac a b

%%
[avg_est_nfl_naa, avg_est_nfl_pch, avg_est_nfl_glx, avg_est_nfl_ins, err_nfl_naa, err_nfl_pch, err_nfl_glx, err_nfl_ins] = testnfl(X_tr, model_naa, model_pch, model_glx, model_ins, reg_num);
fprintf('Mean err naa: %f \n', mean(err_nfl_naa))
fprintf('Mean err pch: %f \n', mean(err_nfl_pch))
fprintf('Mean err glx: %f \n', mean(err_nfl_glx))
fprintf('Mean err ins: %f \n', mean(err_nfl_ins))

%%
[avg_est_csi_naa, avg_est_csi_pch, avg_est_csi_glx, avg_est_csi_ins, err_csi_naa, err_csi_pch, err_csi_glx, err_csi_ins] = testcsi(X_tr, model_naa, model_pch, model_glx, model_ins, reg_num);
fprintf('Mean err naa: %f \n', mean(err_csi_naa))
fprintf('Mean err pch: %f \n', mean(err_csi_pch))
fprintf('Mean err glx: %f \n', mean(err_csi_glx))
fprintf('Mean err ins: %f \n', mean(err_csi_ins))

%%
%% Run processing pipeline

for i = 2%:4
     
     subject_info = Volunteer.(sprintf('Sub%d',i));
     snr_info.(sprintf('Volunteer%d',i)) = snr_rf_test(subject_info, model_naa, model_pch, model_glx, model_ins, reg_num);
end
                
%save('Mar292018_Volunteer_SNR_Results.mat', 'Volunteer', 'snr_info')
%%
a = 1;
b= samp_tot;
nTrees = 80;

for i = 1:reg_num
  
    X_train_obl{i} = X_tr_obl(train_samp(a:b), :);
   
     if b<(data_sz+1)
    a = a + samp_tot;
    b = a + samp_tot - 1;
    reg_count = i;
    end
end  


for i = 1:reg_num
   model_naa{i} =  regRF_train(double(X_train_obl{i}), double(Y_train_naa{i}), nTrees, 8); 
   fprintf('\n UV Naa count: %f\n', i)
   
   model_pch{i} =  regRF_train(double(X_train_obl{i}), double(Y_train_pch{i}), nTrees, 8); 
   fprintf('\n UV PCh count: %f\n', i)
   
   model_glx{i} =  regRF_train(double(X_train_obl{i}), double(Y_train_glx{i}), nTrees, 8); 
   fprintf('\n UV Glx count: %f\n', i)
   
   model_ins{i} =  regRF_train(double(X_train_obl{i}), double(Y_train_ins{i}), nTrees, 8); 
   fprintf('\n UV Ins count: %f\n', i)
   
   model_t2{i} =  regRF_train(double(X_train_obl{i}), double(Y_train_t2{i}), nTrees, 8); 
   fprintf('\n UV T2 count: %f\n', i)
   
   model_mac{i} =  regRF_train(double(X_train_obl{i}), double(Y_train_mac{i}), nTrees, 8); 
   fprintf('\n UV Mac count: %f\n', i)
   
    Est_naa_obl{i} = regRF_predict(double(X_test_obl), model_naa{i});
    Est_pch_obl{i} = regRF_predict(double(X_test_obl), model_pch{i});
    Est_t2_obl{i} = regRF_predict(double(X_test_obl), model_t2{i});
    Est_glx_obl{i} = regRF_predict(double(X_test_obl), model_glx{i});
    Est_ins_obl{i} = regRF_predict(double(X_test_obl), model_ins{i});
    Est_mac_obl{i} = regRF_predict(double(X_test_obl), model_mac{i});
     
    avg_est_naa_obl(:,i) = Est_naa_obl{i};
    avg_est_pch_obl(:,i) = Est_pch_obl{i};
    avg_est_glx_obl(:,i) = Est_glx_obl{i};
    avg_est_ins_obl(:,i) = Est_ins_obl{i};
    avg_est_t2_obl(:,i) = Est_t2_obl{i};
    avg_est_mac_obl(:,i) = Est_mac_obl{i};
end

save('Oblique_Training_sim_27012018.mat', 'X_tr_obl','X_train_obl','dat_ind','train_samp', 'model_glx','model_ins','model_naa','model_pch','Y_naa_ts','Y_pch_ts','Y_glx_ts','Y_ins_ts')
clearvars model_naa model_pch model_t2 model_glx model_ins model_mac a b

