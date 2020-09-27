// NOT USED
// TODO Figure out what it does

function spec, mag, params = loadData( paths )

for i = 1:length(paths)
	if strcmp(paths(i),'parameters')
		params = load(paths(i))
	elif strcmp(paths(i),'magnitude')
		mag = load(paths(i))
	elif strcmp(path(i),'spectra')
		spec = load(paths(i))
	else
		disp('Error: expected file names not found')
	end
end
