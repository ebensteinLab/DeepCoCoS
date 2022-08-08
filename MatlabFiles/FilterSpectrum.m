function Filters=FilterSpectrum(plot_flag,notch4or5band)
%inputs: plot_flag: 1 for plotting
%        notch5: 1 for 4 notch filter
%                0 fo 5 band filter
path=fullfile(pwd,"FilterSpectra");
if nargin==0
    plot_flag=1;
    notch4or5band=1;
elseif nargin==1
    notch4or5band=1;
end
Notch785=importdata(fullfile(path,'785Notch.txt'));
if notch4or5band
Notch4c=importdata(fullfile(path,'4NotchFilter.txt'));
else
Notch4c=importdata(fullfile(path,'old MBF.txt'));
end
% Di4bands=importdata(fullfile(path,'Di03-R405_488_532_635_Spectrum.txt'));
MBM5c=importdata(fullfile(path,'zt405_488_561_640_785 MBM.txt'));

CombinedSpectrum=MBM5c(MBM5c(:,1)>350&MBM5c(:,1)<1000,:);
% CombinedSpectrum=Di4bands.data(Di4bands.data(:,1)>350&Di4bands.data(:,1)<1000,:);

CombinedSpectrum(:,2)=CombinedSpectrum(:,2).*Notch4c.data(ismember(Notch4c.data(:,1),CombinedSpectrum(:,1)),2);
CombinedSpectrum(ismember(CombinedSpectrum(:,1),Notch785.data(:,1)),2)=CombinedSpectrum(ismember(CombinedSpectrum(:,1),Notch785.data(:,1)),2).*Notch785.data(ismember(Notch785.data(:,1),CombinedSpectrum(:,1)),2);
if plot_flag
figure()
plot(Notch785.data(:,1),Notch785.data(:,2),'-r','LineWidth',2);
hold on
plot(Notch4c.data(:,1),Notch4c.data(:,2),'-b','LineWidth',2);
plot(MBM5c(:,1),MBM5c(:,2),'-g','LineWidth',2);
plot(CombinedSpectrum(:,1),CombinedSpectrum(:,2),'-k','LineWidth',2);
xlim([350 1000])
ylim([0 1])
end
[BWmap,n]=bwlabeln(CombinedSpectrum(:,2)>0.4);
SpectralWindow=zeros(n,2);
Filters=zeros(2,n);
for i=1:n
SpectralWindow(i,1)=CombinedSpectrum(find(BWmap(:)==i,1,'first'),1);
SpectralWindow(i,2)=CombinedSpectrum(find(BWmap==i,1,'last'),1);
Filters(1,i)=mean(SpectralWindow(i,:));
Filters(2,i)=diff(SpectralWindow(i,:));

end
end