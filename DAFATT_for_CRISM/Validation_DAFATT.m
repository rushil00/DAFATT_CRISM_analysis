% Usage: 
% Input: the Atomospheric correction IR CRISM data (contains 438 bands,  1.70-2.06um is actually used)
%  output: the blue spectra are the modeled, the orange spectra are pure endmembers

%   Copyright: Honglei Lin (linhonglei9@gmail.com)
%            & Jesse Tarnas (jesse_tarnas@brown.edu)
%  Honglei Lin, J,D.Tarnas, J. F. Mustard, Xia Zhang et al. Dynamic Aperture Factor Analysis/Target Transformation (DAFA/TT)
%  for Serpentine and Mg-Carbonate Mapping on Mars with CRISM Near-Infrared Data. Icarus, 2020.

clear;clc;close all 
prompt='Which target spectrum do you want to check:';% in your target library
N=input(prompt);

%Read the CRISM data 
[filename pathname]=uigetfile('*.img','Select the CRISM Data  (IR data)');
data=freadenvi([pathname filename]);
[Fline,Fsample,Fbands]=size(data);

[filename1 pathname1]=uigetfile('*tiles.img','Select the Tiles Data');
tiles=freadenvi([pathname1 filename1]);
[Fline1,Fsample1,Fbands1]=size(tiles);
data=reshape(data,[Fline*Fsample,Fbands]);

tiles=reshape(tiles,[Fline1*Fsample1,Fbands1]);
values=unique(tiles(:,N));
values=values(2:end); 

load TargetLibrary_paper.mat % Table S2
TargetLibraryRef=TargetLibrary(105:end,N+1);%
TargetLibraryName=TargetLibraryName(N+1);%
TargetLibraryFileName=TargetLibraryFileName(N+1);%
wave=TargetLibrary(105 :end,1);% wavelength

for j=1:length(values)
    
mm=find(tiles(:,N)==j);
data1=data(mm,105:end)';
[kf, NorRMSE,model]=FATT(data1,TargetLibraryRef,TargetLibraryName,wave,'EigNumDM','Hysime');%
TargetLibraryRefNor=TargetLibraryRef./repmat(sum(TargetLibraryRef),[size(data1,1),1]);
modelNor=model./repmat(sum(model),[size(data1,1),1]);

    figure
    set(gcf, 'position', [100 100 380 420]);
    plot(wave,[modelNor,TargetLibraryRefNor],'linewidth',1.5)
      set(gca,'xlim',[1.7 2.6])%
      set(gca,'xtick',[1.7:0.1:2.6])%
      set(gca,'Fontsize',8)

RRMSE{j}=NorRMSE;

end
rr=cell2mat(RRMSE);
[a,b]=sort(rr); 

