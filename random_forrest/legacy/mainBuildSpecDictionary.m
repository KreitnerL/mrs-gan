% NOT USED!

%% Main file to run Basis and Dictionary simulation using ISMRM Workshop basis sets

% Generates a dictionary of spectra with different parameters (metabolite
% concentrations, baseline combinations, lipids, etc.)
% A range of uniformaly distributed values is defined for each parameter
% and a combination of these is taken to generate the spectra
% Note: the range of values includes both diseased and healthy spectra
% along with variations in noise as well.

% (c) Dhritiman Das, 2018 
%(with input from Eduardo Coello)
% TU Munich, Germany

% Update1:Mar 28, 2018 - created separate individual ranges for each
% metabolite, added Remove Baseline post norm evaluation
%%

%clearvars, clc,
doplot = 0;
Ncomb = 3.5;

load('AllBasisSets.mat');
load('Dec10_17_SimulatedBaselines.mat')
%% Parameters
    N=2048;        %Spectral points
    sw=4000;       %Spectral width (Hz)
    dt = 1/sw;     %Dwell time
    Bo=3;          %Magnetic Field Strength (Tesla)
    te = 35;       %PRESS echo time 
    Ns = 2048;
    
    hz = linspace(-sw/2,sw/2,N);    
    t = dt:dt:dt*N;
    ppm = linspace(4.7-16,4.7+16,N); ppm = ppm(end:-1:1);
    ppmU = linspace(4.7-16,4.7+16,Ns); ppmU = ppmU(end:-1:1);
    
%% Basis Simulation
%Note: specws is of length 23. the list is 25. To correspond with an
%element in the list, add +2 to the specws index for the corresponding
%metabolite

    specAce = (specws(:, 1));
    specAla = (specws(:, 2));
    specAsc = (specws(:, 3));
    specAsp = (specws(:, 4));
    specGaba = (specws(:, 6));
    specGPC = (specws(:, 7));
    specGSH = (specws(:, 8));
    specGlc = (specws(:, 9));
    specGln = (specws(:, 10));
    specGlu = (specws(:, 11));
    specGly = (specws(:, 12));
    specLac = (specws(:, 14));
    specNaag = (specws(:, 18));
    specPcr = (specws(:, 20));
    specPe = (specws(:, 21));
    specTau = (specws(:, 22));
    specSIns = (specws(:, 23));
    
    specCh = (specws(:,19)); %specws(:,7) + 
    specCr = (specws(:,5)); % + specws(:,20);
    specNaa = (specws(:,17));
    specLip = (specws(:,15));
    specMac = (specws(:,16));
%     specLac = (specws(:, 14));
     specGlx =  (specws(:, 10)) + (specws(:, 11));
     specIns = (specws(:, 13));
    
    %% Extract FIDs and Spectra
    t2 = 0.1;
    
    fidAce = ifft(fftshift(specAce)).* exp(-t/t2).';
    fidAla = ifft(fftshift(specAla)).* exp(-t/t2).';
    fidAsc = ifft(fftshift(specAsc)).* exp(-t/t2).';
    fidAsp = ifft(fftshift(specAsp)).* exp(-t/t2).';
    fidGaba = ifft(fftshift(specGaba)).* exp(-t/t2).';
    fidGPC = ifft(fftshift(specGPC)).* exp(-t/t2).';
    fidGSH = ifft(fftshift(specGSH)).* exp(-t/t2).';
    fidGlc = ifft(fftshift(specGlc)).* exp(-t/t2).';
    fidGln = ifft(fftshift(specGln)).* exp(-t/t2).';
    fidGlu = ifft(fftshift(specGlu)).* exp(-t/t2).';
    fidGly = ifft(fftshift(specGly)).* exp(-t/t2).';
    fidLac = ifft(fftshift(specLac)).* exp(-t/t2).';
    fidNaag = ifft(fftshift(specNaag)).* exp(-t/t2).';
    fidPcr = ifft(fftshift(specPcr)).* exp(-t/t2).';
    fidPe = ifft(fftshift(specPe)).* exp(-t/t2).';
    fidTau = ifft(fftshift(specTau)).* exp(-t/t2).';
    fidSIns = ifft(fftshift(specSIns)).* exp(-t/t2).';
    
    
    fidCh = ifft(fftshift(specCh)).* exp(-t/t2).';
    fidCr = ifft(fftshift(specCr)).* exp(-t/t2).';
    fidNaa = ifft(fftshift(specNaa)).* exp(-t/t2).';
    fidLip = ifft(fftshift(specLip)).* exp(-t/t2).';
    fidMac = ifft(fftshift(specMac)).* exp(-t/t2).';
%     fidLac = ifft(fftshift(specLac)).* exp(-t/t2).';
     fidGlx = ifft(fftshift(specGlx)).* exp(-t/t2).';
     fidIns = ifft(fftshift(specIns)).* exp(-t/t2).';
    
%     fidCh = RemoveZeroOrderPhase(fidCh ./ abs(fidCh(1,:)));
%     fidCr = RemoveZeroOrderPhase(fidCr ./ abs(fidCr(1,:)));
%     fidNaa = RemoveZeroOrderPhase(fidNaa ./ abs(fidNaa(1,:)));
%     fidLip = RemoveZeroOrderPhase(fidLip ./ abs(fidLip(1,:)));
%     fidMac = RemoveZeroOrderPhase(fidMac ./ abs(fidMac(1,:)));

    fidCh = RemoveZeroOrderPhase(fidCh);% ./ abs(fidCh(1,:))); %% NO Normalization to be compatible with LCModel ratios
    fidCr = RemoveZeroOrderPhase(fidCr);% ./ abs(fidCr(1,:)));
    fidNaa = RemoveZeroOrderPhase(fidNaa);% ./ abs(fidNaa(1,:)));
    fidLip = RemoveZeroOrderPhase(fidLip);% ./ abs(fidLip(1,:)));
    fidMac = RemoveZeroOrderPhase(fidMac);% ./ abs(fidMac(1,:)));
%     fidLac = RemoveZeroOrderPhase(fidLac);
     fidGlx = RemoveZeroOrderPhase(fidGlx);
     fidIns = RemoveZeroOrderPhase(fidIns);
    
    specCh = fftshift(fft(fidCh));
    specCr = fftshift(fft(fidCr));
    specNaa = fftshift(fft(fidNaa));
    specLip = fftshift(fft(fidLip));
    specMac = fftshift(fft(fidMac));
%     specLac = fftshift(fft(fidLac));
     specGlx = fftshift(fft(fidGlx));
     specIns = fftshift(fft(fidIns));
    
    
     % Select baselines - only from base 5 (Base1-4 have been generated
     % using cosine signal variations. Base 5 onwards have been extracted
     % from in-vivo spectra
     
     base1  = (sim_baseline{1,1}.base)./max(sim_baseline{1,1}.base); %baseline 5
     base2  = (sim_baseline{1,6}.base)./max(sim_baseline{1,6}.base); %baseline 6
     base3  = (sim_baseline{1,7}.base)./max(sim_baseline{1,7}.base); %baseline 7
     base4  = (sim_baseline{1,8}.base)./max(sim_baseline{1,8}.base); %baseline 8
     base5  = (sim_baseline{1,9}.base)./max(sim_baseline{1,9}.base); %baseline 9
     
     
     base1 = base1.';
     base2 = base2.';
     base3 = base3.';
     base4 = base4.';
     base5 = base5.';
%% Simulate Standard Spectrum
    aa = 0.3; bb = 1; cc = 1.3; dd = 1; ee = 0.5; ff = 0.3; gg = 0.5; hh = 0.7;  
    
    stdFid = aa*fidCh + bb*fidCr + cc*fidNaa ;%+ dd*fidLip + ee*fidMac + ff*fidLac + gg*fidGlx + hh*fidIns;
    stdFid = RemoveZeroOrderPhase(stdFid);% ./ abs(stdFid(1,:)));
    %Add Noise
    noise = 1 * max(abs(stdFid)) * (rand(size(stdFid))-0.5);
    stdFid = stdFid + noise;
    %Spectrum
    stdSpec = (fftshift(fft(stdFid)));
        
    
    %%
    figure(101), clf, plot(ppm,real(RemoveBaseline(cat(2, aa*specCh, bb*specCr, cc*specNaa))), 'Linewidth', 2);%dd*specLip, ee*specMac, ff*specLac, gg*specGlx, hh*specIns))),'LineWidth',2);
    set(gca,'Xdir','reverse'); xlim([0.2 5.0]); drawnow; hold on;
    plot(ppm,real(RemoveBaseline(stdSpec)),'LineWidth',2.5); hold off;

%% Simulate Dictionary
   
naa_min_range = 0.05; % Min metabolite range for Naa, Choline
naa_max_range = 3.5; % Max metabolite range for Naa, Choline
pch_min_range = 0.05; % Min metabolite range for Naa, Choline
pch_max_range = 1.5; % Max metabolite range for Naa, Choline
glx_min_range = 0.1; % Metabolite range for Glx, Ins
glx_max_range = 5; % Metabolite range for Glx, Ins
ins_min_range = 0.01; % Metabolite range for Glx, Ins
ins_max_range = 2; % Metabolite range for Glx, Ins
cr_range = 1;
B = cr_range;
t2Vector_min = 0.05;
t2Vector_max = 0.7;
mac_min_range = 1;
mac_max_range = 4;
lip_min_range = 1;
lip_max_range = 4;
snr_min_range = 10;
snr_max_range = 50;
baseline_min_range = 0;
baseline_max_range = 6;

%% Dictionary Initialization

    totalEntries = 1000000; 
     %   totalEntries = length(aVector)*length(bVector)*length(cVector)*length(dVector);

    dictionary = single(zeros( totalEntries, Ns));
    dict_obl = single(zeros( totalEntries, 12));
    dictionaryReduced = single(zeros(totalEntries, 263));
    labels = single(zeros(totalEntries, 14));
    rlabels = single(zeros(totalEntries, 14));
    
clearvars N sw dt Bo te Ns Ncomb stdFid stdSpec aa bb cc dd ee ff gg hh;
    entry = 0;

    %%
    for i = 1 : totalEntries
                            entry = entry + 1;
                            random_gen(i) = rng;
                            
                             fprintf('\n entry: %f\n', entry)
                           
                          A = (pch_max_range - pch_min_range).*rand + pch_min_range; % Choline   
                          C = (naa_max_range - naa_min_range).*rand + naa_min_range; % NAA
                          
                          D = (mac_max_range - mac_min_range).*rand + mac_min_range; % Mac
                          L = (lip_max_range - lip_min_range).*rand + lip_min_range; % Lipids
                          
                          H = (glx_max_range - glx_min_range).*rand + glx_min_range; % Glx
                          I = (ins_max_range - ins_min_range).*rand + ins_min_range; % Ins
                          
                          t2 = (t2Vector_max - t2Vector_min).*rand + t2Vector_min; % t2
                          
                          base_scale = randi([baseline_min_range baseline_max_range],1,5);
                          
                          E = randi([snr_min_range snr_max_range],1,1); %Noise
                          
                          %%
                            % Add scaled signals
                            %fidSum = ((A*fidCh + B*fidCr + C*fidNaa + D*fidLip + E*fidMac).');
                            fidSum = ((A*fidCh + B*fidCr + C*fidNaa + D*fidMac+ L*fidLip + H*fidGlx + I*fidIns));% + L*fidLip + G*fidLac + H*fidGlx + I*fidIns));

                           
                            
                            % Add Line Broadening (T2 effect)
                            fidSumT2 = fidSum .* exp(-t/t2).';

                            % Normalize Amplitude
                            fidSumNorm = RemoveZeroOrderPhase(fidSumT2 ./ abs(fidSumT2(1,:)));
                          %   fidSumNorm = RemoveZeroOrderPhase(fidSum ./ abs(fidSum(1,:)));
                           
                          % Add % Noise
                           
                            noise = E * 0.01 * max(abs(fidSumNorm)) * (rand(size(fidSumNorm))-0.5);
                            fidSumNoise = fidSumNorm + noise; 

                            % Recover Spectrum
                            specSummed = fftshift(fft(fidSumNoise));
                            
                            % Add Baseline - non linear to give wavey
                            % effect
                            
                            base_summed =  base_scale(1)*base1 + base_scale(2)*base2 + base_scale(3)*base3 + base_scale(4)*base4 + base_scale(5)*base5;
                            
                            specSummed(1050:1312, :) = specSummed(1050:1312,:) + base_summed; 

                            
                            %% Remove Baseline offset
                            
                          % specSummed = RemoveBaseline(specSummed);
                            
                            
                            %% Normalize Spectra by dividing by the norm of the area under the 3 major peaks
                            %specSummed_cp = specSummed;
                            
                            spec_norm = specSummed./norm(abs(specSummed(1084:1253, :)));
                      %  spec_norm = RemoveBaseline(spec_norm);
                           
                          specSummed_obl = [dot(spec_norm(1050:1312), specCh(1050:1312)), dot(spec_norm(1050:1312), specCr(1050:1312)), dot(spec_norm(1050:1312), specNaa(1050:1312)),...
                                            dot(spec_norm(1050:1312), specGlx(1050:1312)), dot(spec_norm(1050:1312), specIns(1050:1312)),...
                                            dot(spec_norm(1050:1312), specMac(1050:1312)), dot(spec_norm(1050:1312), specLip(1050:1312))....
                                            dot(spec_norm(1050:1312), base1), dot(spec_norm(1050:1312), base2), dot(spec_norm(1050:1312), base3), dot(spec_norm(1050:1312), base4), dot(spec_norm(1050:1312), base5)];
                                        
                                        
                         
                                        %% Add frequency-phase shift effect
%                            fidSummed = ifft(fftshift(specSummed));
%                            fidShift = fidSummed.*(exp(1i*2*pi*((0:2047)*freq_shf + phase_shf))).';
%                            specShifted = fftshift(fft(fidShift));

                     
                            %Normalize
%                             dictionary(:,entry) = single( specSummed / norm(specSummed) );
                          %  dictionary(:,entry) = single( specShifted );
                           dictionary(entry, :) = single(spec_norm.');
                           dictionaryReduced(entry, :) = dictionary(entry, 1050:1312);
                           dict_obl(entry, :) = single(specSummed_obl);
                           labels(entry, :) = single([A B C D E t2 H I L base_scale(1) base_scale(2) base_scale(3) base_scale(4) base_scale(5)]); %D t2 E G H I L freq_shf phase_shf]);
                            rlabels(entry, :) = single([A/B B/B C/B D/B E t2 H/B I/B L/B base_scale(1) base_scale(2) base_scale(3) base_scale(4) base_scale(5)]); %D/B t2 E G/B H/B I/B L/B freq_shf phase_shf]);
                            

                            %if doplot
                            %    figure(1); plot(ppmU, real(dictionary(:,entry))); set(gca,'Xdir','reverse'); 
                                %ylim([-0.1 1.5]); 
                           %     xlim([0.1 5.0]); drawnow; hold on;
                           % end
       
%end
%end
%end
    end
%%
   % labels_list = {'Choline', 'Creatine', 'NAA', 'Mac', 'T2', 'SNR', 'Lactate', 'Glx', 'Inositol', 'Lipids', 'Frequency Shift', 'Phase Shift'};
   labels_list = {'Choline', 'Creatine', 'NAA', 'Mac', 'SNR','T2', 'Glx', 'Inositol', 'Lipid', 'Base1', 'Base2', 'Base3', 'Base4', 'Base5'}; 
 %  dictionaryReduced = dictionary(1050:1312,:);
    %dictionaryReduced_norm = dictionaryReduced./sum(real(dictionaryReduced(75:85, :)));
    save('Mar_29_Dict_RandomSample.mat','dictionary','dictionaryReduced','dict_obl', 'rlabels', 'labels', 'labels_list', 'random_gen')
    clearvars -except dictionary dict_obl dictionaryReduced labels labels_list rlabels
    %% Plot
%    figure(2); plot(ppm, real(dictionary(:,1:1000:end))); 
%    set(gca,'Xdir','reverse');
%    xlim([0.1 5.0]); drawnow; hold on;