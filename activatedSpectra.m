function [ data_real, data_imag ] = activatedSpectra( path1, path2 ) 
%ACTIVATEDSPECTRA Summary of this function goes here
%   Detailed explanation goes here
%
% path1 = '/home/john/Documents/Research/Datasets/UCSF_TUM_MRSI/batch_1/t8771/ucsf/t8771_UCSF_NAA.dcm';
% path2 = '/home/john/Documents/Research/Datasets/UCSF_TUM_MRSI/batch_1/t8771/ucsf/t8771_UCSF_proc.dcm';

%% Identify activated voxels
file = string(path1);
info = dicominfo(file);
image = dicomread(info);

% Code from UCSF file
temp = squeeze(double(image));%reshape(dimX,dimY,dimZ);
map = flip(flip(temp, 1), 2);
baseline = mode(mode(mode(map)));
% [I, J, K] = find(map~=baseline);   % Activated voxels identified
index = find(map~=baseline);

%(*clear file info temp baseline*)

%% Extract activated spectra
file = string(path2);
info = dicominfo(file);

% Code from UCSF file
dimZ = double(info.NumberOfFrames);
dimY = double(info.Columns);
dimX = double(info.Rows);

if isfield(info, 'DataPointColumns')
    dimS = double(info.DataPointColumns);
else
    dimS = 2048;
end

temp = reshape(info.SpectroscopyData,2,dimS,dimX,dimY,dimZ);
tempR = squeeze(temp(1,:,:,:,:));
tempI = squeeze(temp(2,:,:,:,:));
spec = tempR + 1i*tempI;
specProc = flip(flip(permute(spec,[1 3 2 4]),2),3);

% Iterate through the subscripts saving corresponding spectra
% spectra = ones(length(I),dimS);
% for m = 1:length(I)
%     spectra(m,:) = specProc(:, I(m), J(m), K(m));
% end

[I, J, K] = ind2sub(size(map),index);
spectra = ones(length(I),dimS);
for m = 1:length(I)
    spectra(m,:) = specProc(:, I(m), J(m), K(m));
end

% data_real = real(spectra);
% data_imag = imag(spectra);
% data_real = padarray(real(spectra),[0 21],0,'post');
% data_imag = padarray(imag(spectra),[0 21],0,'post');
data_real = real(spectra);
data_imag = imag(spectra);

zerosR = find(data_real==0);
zerosI = find(data_imag==0);

maxR = max(data_real,[],2);
maxI = max(data_imag,[],2);

data_real = (data_real + maxR) ./ (2 * maxR) .* 2 - 1;
data_imag = (data_imag + maxI) ./ (2 * maxI) .* 2 - 1;

if data_real~=data_real
    data_real(zerosR) = 0;
end
if data_imag~=data_imag
    data_imag(zerosI) = 0;
end


% 
% S_real = std(data_real,0,'all','omitnan');
% M_real = mean(data_real,0,'all','omitnan');
% S_imag = std(data_imag,0,'all','omitnan');
% M_imag = mean(data_imag,0,'all','omitnan');
% 
% data_real = (data_real - M_real) / S_real;
% data_imag = (data_imag - M_imag) / S_imag;
% 
% save(path3+'mean_and_std.csv',S_real, S_imag,M_real, M_imag)
% padded_real = padarray(data_real,[0 21],0,'post');
% padded_imag = padarray(data_imag,[0 21],0,'post');

end

