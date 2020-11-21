path_cre = "D:\Datasets\UCSF_TUM_MRSI\batch_1\tar_bundle\data\p41\jasonc\BjoernRandomTrees\data_to_share\for_tum_t8771\ucsf\t8771_UCSF_cre.dcm"
file = string(path_cre);
info = dicominfo(file);
image = dicomread(info);
temp = squeeze(double(image));
map = flip(flip(temp, 1), 2);
baseline = mode(mode(mode(map)));
index = find(map~=baseline);
cre_absolute = map(index);
figure, histogram(cre_absolute), title("cre")

path_cho = "D:\Datasets\UCSF_TUM_MRSI\batch_1\tar_bundle\data\p41\jasonc\BjoernRandomTrees\data_to_share\for_tum_t8771\ucsf\t8771_UCSF_cho.dcm"
file = string(path_cho);
info = dicominfo(file);
image = dicomread(info);
temp = squeeze(double(image));
map = flip(flip(temp, 1), 2);
baseline = mode(mode(mode(map)));
index = find(map~=baseline);
cho_absolute = map(index);
figure, histogram(cho_absolute), title("cho")

path_NAA = "D:\Datasets\UCSF_TUM_MRSI\batch_1\tar_bundle\data\p41\jasonc\BjoernRandomTrees\data_to_share\for_tum_t8771\ucsf\t8771_UCSF_NAA.dcm"
file = string(path_NAA);
info = dicominfo(file);
image = dicomread(info);
temp = squeeze(double(image));
map = flip(flip(temp, 1), 2);
baseline = mode(mode(mode(map)));
index = find(map~=baseline);
NAA_absolute = map(index);
figure, histogram(NAA_absolute), title("NAA")




outlier_cutoff = 10;
not_zero_ind = find(cre_absolute>0);

cho_rel = cho_absolute(not_zero_ind) ./ cre_absolute(not_zero_ind);
normal_ind_cho = find(cho_rel < outlier_cutoff);
figure, histogram(cho_rel(normal_ind_cho)), title("cho/cre"), xlim([0,outlier_cutoff]), ylim([0,200])

NAA_rel = NAA_absolute(not_zero_ind) ./ cre_absolute(not_zero_ind);
normal_ind_NAA = find(NAA_rel < outlier_cutoff);
figure, histogram(NAA_rel(normal_ind_NAA)), title("NAA/cre"), xlim([0,outlier_cutoff]), ylim([0,200])

cni = cho_absolute ./ NAA_absolute;
normal_ind_cni = find(cni < outlier_cutoff);
figure, histogram(cni(normal_ind_cni)), title("cni"), xlim([0,outlier_cutoff]), ylim([0,200])