function [image,p,t,b] = freadenvi(fname)
% freadenvi          - read envi image (V. Guissard, Apr 29 2004)
%
%      Reads an image of ENVI standard type
%     to a [col x line x band] MATLAB array
% image=freadenvi(fname)
% [image,p]=freadenvi(fname)
% [image,p,t]=freadenvi(fname)
% [image,p,t,b] = freadenvi(fname)
%
% INPUT :
%
% fname string giving the full pathname of the ENVI image to read.
%
% OUTPUT :
%
% image----------- c by l by b array containing the ENVI image values organised in
%     c : cols, l : lines and b : bands.
% p   1 by 3 vector that contains (1) the nb of cols, (2) the number.
%     of lines and (3) the number of bands of the opened image.%     
%
% t   string describing the image data type string in MATLAB conventions.
% b   string describing the image data interleave:bsq,bil or bip
%
% 
% Revised by Hu Shunshi,2011.1.3
% can read bsq,bil and bip data interleave.

%%%%%%%%%%%%%

% Parameters initialization
elements={ 'samples' 'lines' 'bands' 'data type' 'interleave'};
d={'uchar' 'int16' 'int32' 'float32' 'float64' 'uint16' 'uint32' 'int64' 'uint64'};
interleave={'bsq' 'bil' 'bip'};
% Check user input
if ~ischar(fname)
    error('fname should be a char string');
end


% 对文件名进行判断，然后选取合适的头文件名称
suffixidx = strfind(fname,'.');
headerfile='';
if length(suffixidx)>0
    headerfile = strcat(fname(1:suffixidx(end)-1),'.hdr');
else
    headerfile = strcat(fname,'.hdr');
end
% Open ENVI header file to retreive s, l, b & d variables
rfid = fopen(headerfile,'r');

% Check if the header file is correctely open
if rfid == -1
    error('Input header file does not exist');
end;

% Read ENVI image header file and get p(1) : nb lines,
% p(2) : nb samples, p(3) : nb bands, t : data type and b:interleave
while 1
    tline = fgetl(rfid);
    if ~ischar(tline), break, end
    [first,second]=strtok(tline,'=');
    % remove whitespace
    first = strtrim(first);
   
    switch first
        case elements(1)
            [f,s]=strtok(second);
            p(1)=str2num(s);
        case elements(2)
            [f,s]=strtok(second);
            p(2)=str2num(s);
        case elements(3)
            [f,s]=strtok(second);
            p(3)=str2num(s);
        case elements(4)
            [f,s]=strtok(second);
            t=str2num(s);
            switch t
                case 1
                    t=d(1);
                case 2
                    t=d(2);
                case 3
                    t=d(3);
                case 4
                    t=d(4);
                case 5
                    t=d(5);
                case 12
                    t=d(6);
                case 13
                    t=d(7);
                case 14
                    t=d(8);
                case 15
                    t=d(9);
                otherwise
                    error('Unknown image data type');
            end
        case elements(5)
            [f,b]=strtok(second);   
            b=strtrim(b);
    end
end
fclose(rfid);

t=t{1,1};
% Open the ENVI image and store it in the 'image' MATLAB array
disp([('Opening '),(num2str(p(2))),('lines x '),(num2str(p(1))),('cols x '),(num2str(p(3))),('bands')]);
disp([('of type '), (t), (' image...')]);
fid=fopen(fname);
image=fread(fid,t);

switch b
    case interleave(1)
%         bsq
        image=reshape(image,[p(1),p(2),p(3)]);
        for dimen=1:p(3)
            image1(:,:,dimen)=image(:,:,dimen)';
        end
        image=image1;
    case interleave(2)        
%         bil
        image1=zeros(p(2),p(1),p(3));
        count=1;
        for row =1:p(2)
            for dimen=1:p(3)       
               image1(row,:,dimen) =image((count-1)*p(1)+1:count*p(1));
               count=count+1;
            end
        end
        image=image1;
    case interleave(3)
%         bip
        image1=zeros(p(2),p(1),p(3));
        count=1;
        for row=1:p(2)
            for col=1:p(1)
                image1(row,col,:)=image((count-1)*p(3)+1:count*p(3));
                count=count+1;
            end
        end 
        image=image1;
    otherwise
        error('Unknown image data interleave');
end

fclose(fid);