% Convert DCM file to matlab file. Called by preprocess_UCSF.py
% Matlab file will later be loaded by the DataLoader

function [ data_real, data_imag ] = activatedSpectra( path1, path2 ) 

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

data_real = real(spectra);
data_imag = imag(spectra);

maxR = max(data_real,[],2);
maxI = max(data_imag,[],2);

data_real = (data_real + maxR) ./ (2 * maxR) .* 2 - 1;
data_imag = (data_imag + maxI) ./ (2 * maxI) .* 2 - 1;

if data_real~=data_real
    data_real(data_real==0) = 0;
end
if data_imag~=data_imag
    data_imag(data_imag==0) = 0;
end

end

