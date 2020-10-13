
function [avg_est_csi_naa, avg_est_csi_pch, avg_est_csi_glx, avg_est_csi_ins, err_naa, err_pch, err_glx, err_ins] = testcsi(X_tr, model_naa, model_pch, model_glx, model_ins, reg_num)


load('02_16_reconCsi2d-LCModelResults.mat')

%%
N=384;
sw=2000;
hz = linspace(-sw/2,sw/2,N);
ppm = hz/128 + 4.7;
ppm = ppm(end:-1:1);
%reconSpec = permute(reconstructedSpec, [2 3 1]);
%reconSpec = reshape(reconstructedSpec, 384, 352); 
reconSpec = reconstructedSpec(:, 3:10, 12:23); 
reconSpec = reshape(reconSpec, 384, 96); 
reconFid = reconstructedFid(:, 3:10, 12:23);
reconFid = reshape(reconFid, 384, 96);

figure, plot(squeeze(ppm),squeeze(real(reconSpec)))
set(gca,'XDir', 'reverse');
xlim([0.2 4.3]);

% LCModel fit takes 4.3 ppm - 0.2 ppm
% 1066+263, 1496-263 (corresponding indices for LCModel fit)
%reconFid = reshape(reconstructedFid,[], 352);

%%
% Water-suppression - added 01/08/2018
[a_arr, lb, freq, spec_sup] =  hsvd(reconFid, 25, [-0.1 0.1], 1, 3);
reconSpec= spec_sup;
%%
% PPM Shift and scaling
csi_shift = interp1(ppm(202:303), reconSpec(203:304,:), linspace(4.3,0.2, 263)); % For CSI: linspace(4.3, 0.2, 431), for Sim Dict: linspace(4.3, 0.2, 263)
%figure, plot(real(csi_shift(:, 142)))

%%
csi_norm = zeros(size(csi_shift)); 
%csi_shift = RemoveBaseline(csi_shift);
 
          rxx = sum(real(A(:)).^2);
    for i = 1: size(csi_shift,2)
      %  csi_norm(:,i) = csi_shift(:,i) ./ norm(real(csi_shift(55:173,i)));%norm(real(csi_shift(55:173,i)));
    % csi_norm(:,i) = csi_shift(:,i) ./ norm(abs(csi_shift(55:173,i)));
       % csi_norm(:,i) = (csi_shift(:,i)).* (norm(abs(A(60:160,:)))./ norm(abs(csi_shift(55:173,i))));

      %   rxy = sum(conj(real(A(:, :))).*real(csi_shift(:,i)));
         %scl = rxy/rxx;
         scl = norm(abs(A(60:160,:)));
         csi_norm(:,i) = csi_shift(:,i) .* scl;
    end
    
   %%
   % Visualization of bias
   x = linspace(1,263, 263).';
   meancsi = mean(real(csi_norm), 2);
   meanplusstd  = real(meancsi) +  std(abs(csi_norm), 0, 2);
   meanminusstd  = real(meancsi) -  std(abs(csi_norm), 0, 2);
   figure, fill( [x fliplr(x)],  [ meanplusstd fliplr(meanminusstd)], 'k')%, hold on
alpha(.25);
plot(x, meancsi, 'k', 'LineWidth', 2)
plot(x, (meanplusstd), 'k')
plot(x, (meanminusstd), 'k')
   
    %%
   % csi_norm = RemoveBaseline(csi_norm);
    csi_norm = permute(csi_norm, [2 1]);
    csi_norm = reshape(csi_norm, 11, 32, 263);
    
    
    
    X_test_csi = (csi_norm(3:10, 12:23, :));
    X_test_csi = reshape(X_test_csi, 96, 263);

    %figure, plot(real(X_test_csi(20, :))), hold on, plot(real(dictionaryReduced(5, :)))%, legend(['CSI', 'Training'])
    %figure, plot(real(X_test_csi(40, :))), hold on, plot(real(dictionaryReduced(8, :)))%, legend('CSI', 'Training')
   % figure, plot(abs(real(X_test_csi(20, :)) - real(dictionaryReduced(5, :)))), title('Diff: CSI(N) 20, Dict(B+N) 5')
   %  figure, plot(abs(real(X_test_csi(40, :)) - real(dictionaryReduced(8, :)))), title('Diff: CSI(N) 40, Dict(B+N) 8')
%  figure, plot(real(X_test_csi(20, :))), hold on, plot(real(X_tr(50000, :))), legend('CSI', 'Training')
%     figure, plot(real(X_test_csi(20, :))), hold on, plot(real(X_tr(130000, :))), legend('CSI', 'Training')
    
%Y_nfl_ts_cr = lcmOutput(1,:).';
%Y_tst_cr = reshape(relconcentration(3:10, 12:23,1),[],1);
Y_tst_pch = reshape(relconcentration(3:10,12:23,2), [],1);
Y_tst_naa = reshape(relconcentration(3:10,12:23,3), [],1);
Y_tst_mI = reshape(relconcentration(3:10,12:23,4), [],1);
Y_tst_glx = reshape(relconcentration(3:10,12:23,5), [],1);

%%

%reg_num =  size(model_naa,2);

for i = 1:reg_num
        Est_naa{i} = regRF_predict(double(X_test_csi), model_naa{i});
        fprintf('\n NAA Testing Complete\n')

        Est_pch{i} = regRF_predict(double(X_test_csi), model_pch{i});
        fprintf('\n PCh Testing Complete\n')

        Est_glx{i} =  regRF_predict(double(X_test_csi), model_glx{i});
        fprintf('\n Glx Testing Complete\n')

        Est_ins{i} = regRF_predict(double(X_test_csi), model_ins{i});
        fprintf('\n mI Testing Complete\n')
%
 %       Est_mac{i} = regRF_predict(double(X_test), model_mac{i});
        fprintf('\n Mac Testing Complete for i: %f\n', i)
        
        avg_est_naa(:,i) = Est_naa{i};
        avg_est_pch(:,i) = Est_pch{i};
        avg_est_glx(:,i) = Est_glx{i};
        avg_est_ins(:,i) = Est_ins{i};
  %      avg_est_mac(:,i) = Est_mac{i};
end

%%

avg_est_csi_naa = mean(avg_est_naa, 2);
avg_est_csi_pch = mean(avg_est_pch, 2);
avg_est_csi_glx = mean(avg_est_glx, 2);
avg_est_csi_ins = mean(avg_est_ins, 2);


err_naa = (abs(avg_est_csi_naa - Y_tst_naa))./(abs(Y_tst_naa));
err_pch = (abs(avg_est_csi_pch - Y_tst_pch))./(abs(Y_tst_pch));
err_glx = (abs(avg_est_csi_glx - Y_tst_glx))./(abs(Y_tst_glx));
err_ins = (abs(avg_est_csi_ins - Y_tst_mI))./(abs(Y_tst_mI));


figure, boxplot([err_naa, err_pch, err_glx, err_ins],'Labels',{'NAA', 'PCh', 'Glx', 'Ins'}, 'Notch','on'), ylim([0 1.2])

%%
% load('C:\Users\DhritimanTUM\Documents\Python_DeepLearning_MRS\predictions_csi_2002.mat')
% dl_err_naa = (abs(Y_naa_pred.' - Y_tst_naa))./(abs(Y_tst_naa));
% dl_err_pch = (abs(Y_pch_pred.' - Y_tst_pch))./(abs(Y_tst_pch));
% dl_err_glx = (abs(Y_glx_pred.' - Y_tst_glx))./(abs(Y_tst_glx));
% dl_err_ins = (abs(Y_ins_pred.' - Y_tst_mI))./(abs(Y_tst_mI));
% 
% %%
% ml_err_naa = err_naa;
% ml_err_pch = err_pch;
% ml_err_glx = err_glx;
% ml_err_ins = err_ins;
% 
% %%
% 
% position_1 = 1:1:4;
% position_2 = 1.3:1:4.3;
% f = figure,
% box_ml = boxplot([ml_err_naa, ml_err_pch, ml_err_glx, ml_err_ins],'Labels',{'ML-NAA   DL-NAA', 'ML-PCH  DL-PCh', 'ML-Glx  DL-Glx', 'ML-Ins  DL-Ins'}, 'Notch','on', 'colors','b','positions',position_1, 'width', 0.2), ylim([0 1])
% hold on;
% box_dl = boxplot([dl_err_naa, dl_err_pch, dl_err_glx, dl_err_ins],'Labels',{'ML-NAA   DL-NAA', 'ML-PCH  DL-PCh', 'ML-Glx  DL-Glx', 'ML-Ins  DL-Ins'}, 'Notch','on', 'colors','r','positions',position_2, 'width', 0.2), ylim([0 2])

%%

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


% % %     computeBlandAltman(avg_est_csi_naa, Y_tst_naa, 1, 'NAA');
% % %     computeBlandAltman(avg_est_csi_pch, Y_tst_pch, 2, 'Cho');
% % %     computeBlandAltman(avg_est_csi_ins, Y_tst_mI, 3, 'Ins');
% % %     computeBlandAltman(avg_est_csi_glx, Y_tst_glx, 4, 'Glx');

 %%
 %======================================================================================================================
 % Regression Plots
 % X-axis = LCModel estimates
 % Y-axis = Regression estimates
 %======================================================================================================================
 
% % %     h(1)= figure; plotregression(Y_tst_naa, avg_est_csi_naa,  'Regression NAA');
% % %     h(2)= figure; plotregression(Y_tst_pch, avg_est_csi_pch,  'Regression Cho');
% % %     h(3)= figure; plotregression(Y_tst_mI, avg_est_csi_ins,  'Regression Ins');
% % %     h(4)= figure; plotregression(Y_tst_glx, avg_est_csi_glx, 'Regression Glx');
% % %     savefig(h, 'Regression_CSI_Figures.fig');