function status = fwriteenvi(filename,data,wavelength,bandname,wavelengthunit)
 
% Input£º
%   filename   
%   data
%   wavelength: the first column is centeral wavelength;
%               the second column is FWHM
%   wavelengthunit:
%                   1 Nanometers; 2 Micrometers£»
%                   3 Wavenumber; 4 Unknown
%   bandname: data type is cell
% Output£º
%   status :   -1£ºfail£¬1£ºSuccess 
% Author£ºHu Shunshi
% Modifier: Honglei Lin 

% Check user input
if ~ischar(filename)
    error('filename should be a char string');
end
[m,n,d] = size(data);
if m==0 || n==0 ||isempty(data)
    error('data is empty');
end
% check wavelength unit
if(~isempty(wavelengthunit) && (wavelengthunit>4 ||wavelengthunit<1))
    error('wavelength unit is not correct');
end

[pathstr, name, ext] = fileparts(filename);


if length(pathstr)==0
    enviheadfile = [name '.hdr'];
else
    enviheadfile = [pathstr '\' name '.hdr'];
end

fidhead = fopen(enviheadfile,'w+');
if fidhead <0
    error('open head file error');
end

datatype={'bit8' 'uint8' 'int16' 'int32' 'float32' 'float64' 'uint16' 'uint32' 'int64' 'uint64' 'double'};
datatypeid={'1' '1' '2' '3' '4' '5' '12' '13' '14' '15' '5' };
interleave={'bsq' 'bil' 'bip'};
unit ={ 'Nanometers' 'Micrometers' 'Wavenumber' 'Unknown'};

pos = strmatch(class(data),datatype);
if isempty(pos)
    error('unsupported data type');
end

fprintf(fidhead,'%s\n%s\n','ENVI', 'description = {');
fprintf(fidhead,'%s\n',['    Create New File Result [' datestr(fix(clock)) ']}']);
fprintf(fidhead,'samples = %d\nlines = %d\nbands   = %d\n',n,m,d);
fprintf(fidhead,'%s\n%s\n','header offset = 0', 'file type = ENVI Standard');


fprintf(fidhead,'data type = %s\ninterleave = %s\n',datatypeid{pos},interleave{1});
fprintf(fidhead,'sensor type = Unknown\nbyte order = 0\nwavelength units = %s\n',unit{wavelengthunit});



if ~isempty(wavelength)   
    if size(wavelength,1) ==1
        wavelength = wavelength(:);
    end
    fprintf(fidhead,'wavelength = {\n');    
    for i =1:size(wavelength,1)-1
        fprintf(fidhead,'%11.6f, ',wavelength(i,1));
        if mod (i,6) ==0
            fprintf(fidhead,'\n');
        end
    end
   
    fprintf(fidhead,'%11.6f}\n',wavelength(end,1));
    
    if size(wavelength,2) == 2
         fprintf(fidhead,'fwhm = {\n');   
         for i =1:size(wavelength,1)-1
            fprintf(fidhead,'%11.6f, ',wavelength(i,2));
            if mod (i,6) ==0
                fprintf(fidhead,'\n');
            end
         end
        fprintf(fidhead,'%11.6f}\n',wavelength(end,2));
    end   
end



% write the bandname to headfile 

if ~isempty(bandname)
 fprintf(fidhead,'\nband names= {\n');    
    for i=1:length(bandname)-1
 fprintf(fidhead,'%s,\n',bandname{i});
    end
    fprintf(fidhead,'%s}\n',bandname{end});
end
fclose(fidhead); 
%--------------------------write to file---------------------------------------
fid =fopen(filename,'wb');
if fid<0
    error('can not open file to write data');
end

for i = 1:d
    fwrite(fid,data(:,:,i)',class(data));
end

fclose(fid);

% -------------------------------------------------------------------------
status = 1;







