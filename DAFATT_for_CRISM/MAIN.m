% Usage: 
% Run this program, then select the folder contains the CRISM images
% The results will be recorded in DAFATTResults folder

%   Copyright: Honglei Lin (linhonglei9@gmail.com)
%            & Jesse Tarnas (jesse_tarnas@brown.edu)
%  Honglei Lin, J,D.Tarnas, J. F. Mustard, Xia Zhang et al. Dynamic Aperture Factor Analysis/Target Transformation (DAFA/TT)
%  for Serpentine and Mg-Carbonate Mapping on Mars with CRISM Near-Infrared Data. Icarus, 2020.

clear;clc;%close all  
%Read the CRISM data 
tic
tr_dir=uigetdir({},'Select the folders containing CRISM I/F data');
D=dir([tr_dir,'\*.img']);
strpos=strfind(tr_dir,'\');
mkdir([tr_dir(1:strpos(end)),'\DAFATTResults']);% create the output folder

FileNum=size(D,1);

parpool('local',4)

for fn=1:FileNum
  
     FileName=strcat(tr_dir,'\',D(fn).name);
     tempfilename=D(fn).name;
    data= freadenvi(FileName);
    [Fline,Fsample,Fbands]=size(data);
    if Fsample==640
        data=data(2:end-1,32:631,105:240);%change the spectral range it as you want 
    else
        data=data(2:end-1,18:314,105:240);%change the spectral range as you want       
    end

[nline,nsample,nband]=size(data);

%load Target Library
load TargetLibrary_paper.mat % Table S2
TargetLibraryRef=TargetLibrary(105:end,2:end);%
TargetLibraryName=TargetLibraryName(2:end);
TargetLibraryFileName=TargetLibraryFileName(2:end);
n=size(TargetLibraryRef,2);% the spectra number in target library  
wave=TargetLibrary(105:end,1);% wavelength

a=[6,8,5,7,10];
b=[8,6,10,7,5]; % this is the dynamic apertures, you can add any aperture as you want, a and b should be coupled 


parfor window=1:size(a,2)
    fprintf('processing the %dth window of %d windows of %dth file of %d files\n',window,size(a,2),fn,FileNum)    

     detect(nline,nsample,n)=0;
for i=1:nline-a(window)+1
    for j=1:nsample-b(window)+1
        data1=reshape(data(i:i+a(window)-1,j:j+b(window)-1,:),[a(window)*b(window),nband])';
        [kf, NorRMSE,model]=FATT(data1,TargetLibraryRef,TargetLibraryName,wave,'EigNumDM','Hysime');% using Hysime to determine eigenvectors, see FATT function 
  % record the windows, which normalized RMSE lower than 1.5*10-4
       for num=1:n
           if NorRMSE(num)<=1.5e-4
               detect(i:i+a(window)-1,j:j+b(window)-1,num)=1;
           end
       end     
    end
end

% detect_square record the results of all minerals of each window  
detect_square(Fline,Fsample,n)=0;
for pp=1:n
 if Fsample==640
detect_square(2:end-1,32:631,pp)=detect(:,:,pp);
 else
detect_square(2:end-1,18:314,pp)=detect(:,:,pp);
 end
end

DETECT{window}=detect_square;

detect=[]; 
detect_square=[];

end 


w1=DETECT{1};
w2=DETECT{2};
w3=DETECT{3};
w4=DETECT{4};
w5=DETECT{5};
inter(Fline,Fsample,n)=0;
for i=1:n
  inter(:,:,i)=w1(:,:,i) & w2(:,:,i) & w3(:,:,i) & w4(:,:,i) & w5(:,:,i);  
end

% write the results to files 
[PATHSTR,NAME,EXT] = fileparts(FileName);
OutputFileName=[tr_dir(1:strpos(end)),'DAFATTResults\',NAME,'_DAFATT.img'];
status = fwriteenvi(OutputFileName,inter,[],TargetLibraryName,[]) 

clear DETECT inter
end
delete(gcp('nocreate'))

toc


