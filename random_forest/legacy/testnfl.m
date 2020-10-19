%%
%%**********************************************
% This script trains on the real synthetic data and tests on the NFL Single-Voxel Spectroscopy dataset.
% The relConcentration variable contains the concentration ratios obtained
% from the LCModel fit of the NFL spectra. This serves as the ground
% truth against which the ML fit is validated.
%==========================================================================
% Version1: Feb 7, 2018
%(c) Dhritiman Das, 2018

%%
function [avg_est_nfl_naa, avg_est_nfl_pch, avg_est_nfl_glx, avg_est_nfl_ins, err_naa, err_pch, err_glx, err_ins] = testnfl(X_tr, model_naa, model_pch, model_glx, model_ins, reg_num)
load('02_16_NflSv-LCModel-Results.mat')

%%

N=2048;
sw=2500;
hz = linspace(-sw/2,sw/2,N);
ppm = hz/128 + 4.7;
ppm = ppm(end:-1:1);
figure, plot(ppm,real(dictionarySVspec))
set(gca,'XDir', 'reverse');
xlim([0.2 4.3]);

% LCModel fit takes 4.3 ppm - 0.2 ppm
% 1066+263, 1496-263 (corresponding indices for LCModel fit)

%% PPM-Shift of the nfl spec

% Since FID is 1024 points
N = 1024;
hz = linspace(-sw/2,sw/2,N);
ppm = hz/128 + 4.7;
ppm = ppm(end:-1:1);


[a_arr, lb, freq, spec_sup] =  hsvd(dictionarySVfid, 25, [-0.1 0.1], 1, 3);
dictionarySVspec= spec_sup;

%nfl_shift = interp1(ppm(1066:1496), dictionarySVspec(1069:1499,:), linspace(4.3,0.2,263), 'linear', 'extrap');
nfl_shift = interp1(ppm(533:748), dictionarySVspec(533:748,:), linspace(4.3,0.2,263), 'linear', 'extrap');
ppm_new = linspace(4.3,0.2,263);
ppm_new = ppm_new(end:-1:1);
%figure, plot(ppm_new, real(nfl_shift(:, 150))), hold on, plot(1:263, real(nfl_shift(:, 150))), legend('with ppm axis', 'w/o')

%% Normalization
 
 %Taking norm under 3 major peaks
 nfl_norm = zeros(size(nfl_shift));
 
 %nfl_shift = RemoveBaseline(nfl_shift);

    
 for i = 1: size(nfl_shift,2)
        nfl_norm(:,i) = nfl_shift(:,i) ./ norm(abs(nfl_shift(:,i)));%norm(abs(nfl_shift(60:163,i)));
    end
    
   

%% Preparing Test set

%nfl_norm = RemoveBaseline(nfl_norm);
nfl_norm = permute(nfl_norm, [2 1]);
X_test = real(nfl_norm);

%Y_nfl_ts_cr = lcmOutput(1,:).';
Y_nfl_ts_pch = lcmOutput(2,:).'; % PCh
Y_nfl_ts_naa = lcmOutput(3,:).'; % NAA
Y_nfl_ts_ins = lcmOutput(4,:).'; % Myo-Inositol
Y_nfl_ts_glx = lcmOutput(5,:).'; % Glx
%Y_nfl_ts_mac = lcmOutput(7,:).'; % Macromolecules

%% Evaluate the test-set

%============================================================================
% v1: Use training set ('Training_sim_27012018.mat')
%==============================================================================


%reg_num =  size(model_naa,2);

for i = 1:reg_num
        Est_naa{i} = regRF_predict(double(X_test), model_naa{i});
        fprintf('\n NAA Testing Complete\n')

        Est_pch{i} = regRF_predict(double(X_test), model_pch{i});
        fprintf('\n PCh Testing Complete\n')

        Est_glx{i} =  regRF_predict(double(X_test), model_glx{i});
        fprintf('\n Glx Testing Complete\n')

        Est_ins{i} = regRF_predict(double(X_test), model_ins{i});
        fprintf('\n mI Testing Complete\n')

     %   Est_mac{i} = regRF_predict(double(X_test), model_mac{i});
      %  fprintf('\n Mac Testing Complete for i: %f\n', i)
        
        avg_est_naa(:,i) = Est_naa{i};
        avg_est_pch(:,i) = Est_pch{i};
        avg_est_glx(:,i) = Est_glx{i};
        avg_est_ins(:,i) = Est_ins{i};
       % avg_est_mac(:,i) = Est_mac{i};
end

%% Error Calculation
avg_est_nfl_naa = mean(avg_est_naa, 2);
avg_est_nfl_pch = mean(avg_est_pch, 2);
avg_est_nfl_glx = mean(avg_est_glx, 2);
avg_est_nfl_ins = mean(avg_est_ins, 2);
%avg_est_nfl_mac = mean(avg_est_mac, 2);

err_naa = (abs(avg_est_nfl_naa - Y_nfl_ts_naa))./(abs(Y_nfl_ts_naa));
err_pch = (abs(avg_est_nfl_pch - Y_nfl_ts_pch))./(abs(Y_nfl_ts_pch));
err_glx = (abs(avg_est_nfl_glx - Y_nfl_ts_glx))./(abs(Y_nfl_ts_glx));
err_ins = (abs(avg_est_nfl_ins - Y_nfl_ts_ins))./(abs(Y_nfl_ts_ins));
%err_mac = (abs(avg_est_nfl_mac - Y_nfl_ts_mac))./(abs(Y_nfl_ts_mac));

figure, boxplot([err_naa, err_pch, err_glx, err_ins],'Labels',{'NAA', 'PCh', 'Glx', 'Ins'}, 'Notch','on'), ylim([0 1])
%%
% DL Error calculation
% load('C:\Users\DhritimanTUM\Documents\Python_DeepLearning_MRS\predictions_nfl_2002.mat')
% dl_err_naa = (abs(Y_naa_pred.' - Y_nfl_ts_naa))./(abs(Y_nfl_ts_naa));
% dl_err_pch = (abs(Y_pch_pred.' - Y_nfl_ts_pch))./(abs(Y_nfl_ts_pch));
% dl_err_glx = (abs(Y_glx_pred.' - Y_nfl_ts_glx))./(abs(Y_nfl_ts_glx));
% dl_err_ins = (abs(Y_ins_pred.' - Y_nfl_ts_ins))./(abs(Y_nfl_ts_ins));
% 
% %%
% % Compute errors
% ml_err_naa = err_naa;
% ml_err_pch = err_pch;
% ml_err_glx = err_glx;
% ml_err_ins = err_ins;
% 
% %%
% position_1 = 1:1:4;
% position_2 = 1.3:1:4.3;
% f = figure,
% box_ml = boxplot([ml_err_naa, ml_err_pch, ml_err_glx, ml_err_ins],'Labels',{'ML-NAA   DL-NAA', 'ML-PCH  DL-PCh', 'ML-Glx  DL-Glx', 'ML-Ins  DL-Ins'}, 'Notch','on', 'colors','b','positions',position_1, 'width', 0.2), ylim([0 1])
% hold on;
% box_dl = boxplot([dl_err_naa, dl_err_pch, dl_err_glx, dl_err_ins],'Labels',{'ML-NAA   DL-NAA', 'ML-PCH  DL-PCh', 'ML-Glx  DL-Glx', 'ML-Ins  DL-Ins'}, 'Notch','on', 'colors','r','positions',position_2, 'width', 0.2), ylim([0 2])
%box_dl = boxplot([dl_err_naa, dl_err_pch, dl_err_glx, dl_err_ins],'Labels',{'ML-NAA   DL-NAA', 'ML-PCH  DL-PCh', 'ML-Glx  DL-Glx', 'ML-Ins  DL-Ins'}, 'Notch','on', 'colors','r','positions',position_2, 'width', 0.3), ylim([0 2])
%hold off

%set(gca,'YLabel','Relative Error rate') 

%%
%==========================================================================================
% Compute Bland-Altman
% Input arguments: Regression estimates; LCModel ground-truth, Metabolite Number (for saving figure) 
%  and Metabolite name ;
%
% Output: 
%        X-Axis: represents the average of the LCModel and Regression estimates
%        Y-Axis: represents the absolute difference between the LCModel and
%        Regression estimates
%==========================================================================================


% % %     computeBlandAltman(avg_est_nfl_naa, Y_nfl_ts_naa, 1, 'NAA');
% % %     computeBlandAltman(avg_est_nfl_pch, Y_nfl_ts_pch, 2, 'Cho');
% % %    % computeBlandAltman(avg_est_nfl_mac, Y_nfl_ts_mac, 3, 'Mac');
% % %     computeBlandAltman(avg_est_nfl_ins, Y_nfl_ts_ins, 4, 'Ins');
% % %     computeBlandAltman(avg_est_nfl_glx, Y_nfl_ts_glx, 5, 'GLX');

 %%
 % Regression Plots
% % % 
% % %     h(1)= figure; plotregression(Y_nfl_ts_naa, avg_est_nfl_naa,  'Regression NAA');
% % %     h(2)= figure; plotregression(Y_nfl_ts_pch, avg_est_nfl_pch,  'Regression Cho');
% % %     h(3)= figure; plotregression(Y_nfl_ts_mac, avg_est_nfl_mac,  'Regression Mac');
% % %     h(4)= figure; plotregression(Y_nfl_ts_ins, avg_est_nfl_ins,  'Regression Ins');
% % %     h(5)= figure; plotregression(Y_nfl_ts_glx, avg_est_nfl_glx, 'Regression Glx');
% % %     savefig(h, 'Regression_NFL_Figures.fig');
%%
end