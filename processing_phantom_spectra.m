% NOT USED
% Example of how to use DCM files.

file = '/home/john/Documents/Research/Datasets/UCSF_TUM_MRSI/phantom spectra/DICOM/phan_lac_fbcomb_sum_cp_cor.dcm';
info = dicominfo(file);
data = info.SpectroscopyData;

dimZ = double(info.NumberOfFrames);
dimY = double(info.Columns);
dimX = double(info.Rows);
dimS = double(info.DataPointColumns);

temp = reshape(info.SpectroscopyData,2,dimS,dimX,dimY,dimZ);
tempR = squeeze(temp(1,:,:,:,:));
tempI = squeeze(temp(2,:,:,:,:));
spec = tempR + 1i*tempI;
specProc = flip(flip(permute(spec,[1 3 2 4]),2),3);


