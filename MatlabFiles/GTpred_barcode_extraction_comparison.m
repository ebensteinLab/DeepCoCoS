%%

% totalCropsNum=[153586,107573,301307,279821];
warning('off', 'MATLAB:mir_warning_maybe_uninitialized_temporary');

parpool;
%% Process multiple crops:

currentFolder = pwd;
path=fullfile(currentFolder,"Data","RawImages");

warning('off', 'MATLAB:mir_warning_maybe_uninitialized_temporary');
wb=waitbar(0,"running sample 1/8...");
[barcodeCountTableL1gt,L1_brc_list_gt,L1gtStack,L1gtCorrectedStack,L1gtmeanBleedParam]=processCrops(fullfile(path,"L1","L1_gtCrops.tif"),[],'HC1 gt',[]);%
save(fullfile(path,'WorkspaceVarsGt.mat'),'barcodeCountTableL1gt','L1gtCorrectedStack','L1gtStack','L1_brc_list_gt','L1gtmeanBleedParam','-v7.3','-nocompression') 

waitbar(1/8,wb,"running sample 2/8...");
[barcodeCountTableL2gt,L2_brc_list_gt,L2gtStack,L2gtCorrectedStack,L2gtmeanBleedParam]=processCrops(fullfile(path,"L2","L2_gtCrops.tif"),[],'UC1 gt',[]);
save(fullfile(path,'WorkspaceVarsGt.mat'),'barcodeCountTableL2gt','L2gtCorrectedStack','L2gtStack','L2_brc_list_gt','L2gtmeanBleedParam','-append','-nocompression') 

waitbar(2/8,wb,"running sample 3/8...");
[barcodeCountTableL3gt,L3_brc_list_gt,L3gtStack,L3gtCorrectedStack,L3gtmeanBleedParam]=processCrops(fullfile(path,"L3","L3_gtCrops.tif"),[],'UC2 gt',[]);
save(fullfile(path,'WorkspaceVarsGt.mat'),'barcodeCountTableL3gt','L3gtCorrectedStack','L3gtStack','L3_brc_list_gt','L3gtmeanBleedParam','-append','-nocompression') 

waitbar(3/8,wb,"running sample 4/8...");
[barcodeCountTableL4gt,L4_brc_list_gt,L4gtStack,L4gtCorrectedStack,L4gtmeanBleedParam]=processCrops(fullfile(path,"L4","L4_gtCrops.tif"),[],'HC2 gt',[]);%
save(fullfile(path,'WorkspaceVarsGt.mat'),'barcodeCountTableL4gt','L4gtCorrectedStack','L4gtStack','L4_brc_list_gt','L4gtmeanBleedParam','-append','-nocompression') 

BleedParams=[L1gtmeanBleedParam,L2gtmeanBleedParam,L3gtmeanBleedParam,L4gtmeanBleedParam];
waitbar(4/8,wb,"running sample 5/8...");
[barcodeCountTableL1pred,L1_brc_listpred,L1predStack,L1predCorrectedStack,L1predmeanBleedParam]=processCrops(fullfile(path,"L1","L1_predCrops.tif"),[],'HC1 pred',geomean(BleedParams(:,1:4),2));
save(fullfile(path,'WorkspaceVarsPred.mat'),'barcodeCountTableL1pred','L1predCorrectedStack','L1predStack','L1_brc_listpred','L1predmeanBleedParam','-append','-nocompression') 

waitbar(5/8,wb,"running sample 6/8...");
[barcodeCountTableL2pred,L2_brc_listpred,L2predStack,L2predCorrectedStack,L2predmeanBleedParam]=processCrops(fullfile(path,"L2","L2_predCrops.tif"),[],'UC1 pred',geomean(BleedParams(:,1:4),2));
save(fullfile(path,'WorkspaceVarsPred.mat'),'barcodeCountTableL2pred','L2predCorrectedStack','L2predStack','L2_brc_listpred','L2predmeanBleedParam','-append','-nocompression') 

waitbar(6/8,wb,"running sample 7/8...");

[barcodeCountTableL3pred,L3_brc_listpred,L3predStack,L3predCorrectedStack,L3predmeanBleedParam]=processCrops(fullfile(path,"L3","L3_predCrops.tif"),[],'UC2 pred',geomean(BleedParams(:,1:4),2));
save(fullfile(path,'WorkspaceVarsPred.mat'),'barcodeCountTableL3pred','L3predCorrectedStack','L3predStack','L3_brc_listpred','L3predmeanBleedParam','-append','-nocompression') 

waitbar(7/8,wb,"running sample 8/8...");

[barcodeCountTableL4pred,L4_brc_listpred,L4predStack,L4predCorrectedStack,L4predmeanBleedParam]=processCrops(fullfile(path,"L4","L4_predCrops.tif"),[],'HC2 pred',geomean(BleedParams(:,1:4),2));
save(fullfile(path,'WorkspaceVarsPred.mat'),'barcodeCountTableL4pred','L4predCorrectedStack','L4predStack','L4_brc_listpred','L4predmeanBleedParam','-append','-nocompression') 

close(wb)
totalCropsNum=[size(L1gtStack,4),size(L2gtStack,4),size(L3gtStack,4),size(L4gtStack,4)];
save('D:\Belinson noBeads\pred files\crops\WorkspaceVarsGt.mat','totalCropsNum','-append','-nocompression') 









%% Comparing the corrected stack

addpath 'C:\Users\admin\Fiji.app\scripts' % Update for your ImageJ2 (or Fiji) installation as appropriate
ImageJ;
imp = copytoImagePlus(L1gtCorrectedStack,'NewName','Corrected_stack');
imp.show();
imp2 = copytoImagePlus(double(L1gtStack),'NewName','Original_stack');
imp2.show();


%% Compare barcode counts
PathnCounter=fullfile(pwd,"Data","nCounter");
nCounterResTable = readtable(fullfile(PathnCounter,'RCC_summary_Lanes1-4.xlsx') ,'ReadVariableNames' ,true,'NumHeaderLines',1,'VariableNamingRule','modify','FileType','spreadsheet','ExpectedNumVariables', 9);
nCounterResTable.CodeClass=categorical(nCounterResTable.CodeClass);
nCounterResTable.Name=categorical(nCounterResTable.Name);
nCounterResTable.Accession=categorical(nCounterResTable.Accession);
nCounterResTable.Barcode=categorical(nCounterResTable.Barcode);
nCounterResTable.BarcodeNumbers=categorical(nCounterResTable.BarcodeNumbers);
nCounterResTable.Properties.VariableNames(6:9)={'HC1 nCounter','UC1 nCounter','UC2 nCounter','HC2 nCounter'};

% % 
barcodeCountTableL1gt.Properties.VariableNames={'BarcodeNumbers','HC1 GT'};
barcodeCountTableL2gt.Properties.VariableNames={'BarcodeNumbers','UC1 GT'};
barcodeCountTableL3gt.Properties.VariableNames={'BarcodeNumbers','UC2 GT'};
barcodeCountTableL4gt.Properties.VariableNames={'BarcodeNumbers','HC2 GT'};
barcodeCountTableL1pred.Properties.VariableNames={'BarcodeNumbers','HC1 Pred'};
barcodeCountTableL2pred.Properties.VariableNames={'BarcodeNumbers','UC1 Pred'};
barcodeCountTableL3pred.Properties.VariableNames={'BarcodeNumbers','UC2 Pred'};
barcodeCountTableL4pred.Properties.VariableNames={'BarcodeNumbers','HC1 Pred'};

combinedCountTable=outerjoin(nCounterResTable,barcodeCountTableL1gt,"Type","left","Keys","BarcodeNumbers","MergeKeys",true);
combinedCountTable=outerjoin(combinedCountTable,barcodeCountTableL2gt,"Type","left","Keys","BarcodeNumbers","MergeKeys",true);
combinedCountTable=outerjoin(combinedCountTable,barcodeCountTableL3gt,"Type","left","Keys","BarcodeNumbers","MergeKeys",true);
combinedCountTable=outerjoin(combinedCountTable,barcodeCountTableL4gt,"Type","left","Keys","BarcodeNumbers","MergeKeys",true);
combinedCountTable=outerjoin(combinedCountTable,barcodeCountTableL1pred,"Type","left","Keys","BarcodeNumbers","MergeKeys",true);
combinedCountTable=outerjoin(combinedCountTable,barcodeCountTableL2pred,"Type","left","Keys","BarcodeNumbers","MergeKeys",true);
combinedCountTable=outerjoin(combinedCountTable,barcodeCountTableL3pred,"Type","left","Keys","BarcodeNumbers","MergeKeys",true);
combinedCountTable=outerjoin(combinedCountTable,barcodeCountTableL4pred,"Type","left","Keys","BarcodeNumbers","MergeKeys",true);

t1=combinedCountTable{:,10:end};

t1(isnan(t1))=0;
combinedCountTable{:,10:end}=t1;
combinedCountTable(end,:)=[];


load(fullfile(PathnCounter,'RCCNameOrder.mat'));
[found,idx]=ismember(RCCorder.Name,combinedCountTable.Name);
combinedCountTable=combinedCountTable(idx,:);

PositiveControlCountTable = sortrows(combinedCountTable(combinedCountTable.CodeClass=="Positive" ,1:end),{'Name'},'ascend');
NegativeControlCountTable = combinedCountTable(combinedCountTable.CodeClass=="Negative",1:end);
HousekeepingCountTable = combinedCountTable(combinedCountTable.CodeClass=="Housekeeping",1:end);
EndogenousCountTable = combinedCountTable(combinedCountTable.CodeClass=="Endogenous",1:end);
%% Write to RCC files

writetable(combinedCountTable(:,[1:3,10]),fullfile(PathnCounter,'RCCfiles','HC1gt.csv'));
writetable(combinedCountTable(:,[1:3,11]),fullfile(PathnCounter,'RCCfiles','UC1gt.csv'));
writetable(combinedCountTable(:,[1:3,12]),fullfile(PathnCounter,'RCCfiles','UC2gt.csv'));
writetable(combinedCountTable(:,[1:3,13]),fullfile(PathnCounter,'RCCfiles','HC2gt.csv'));
writetable(combinedCountTable(:,[1:3,14]),fullfile(PathnCounter,'RCCfiles','HC1pred.csv'));
writetable(combinedCountTable(:,[1:3,15]),fullfile(PathnCounter,'RCCfiles','UC1pred.csv'));
writetable(combinedCountTable(:,[1:3,16]),fullfile(PathnCounter,'RCCfiles','UC2pred.csv'));
writetable(combinedCountTable(:,[1:3,17]),fullfile(PathnCounter,'RCCfiles','HC2pred.csv'));



%% QC checks Bkg threshold and Normalization for Count results:
%Linearity check R^2 values for log2 positive control:
posControlKnown=[ones(6,1),log2([128,32,8,2,0.5,0.125]')];
y=log2(PositiveControlCountTable{:,6:end});
Rsq1=zeros(size(y,2),1);
for sample=1:size(y,2)
    b_samp=posControlKnown\y(:,sample);
    yCalc1=posControlKnown*b_samp;
    Rsq1(sample) = 1 - sum((y(:,sample) - yCalc1).^2)/sum((y(:,sample) - mean(y(:,sample))).^2);
end

%Limit of detetction:
neg_mean=mean(NegativeControlCountTable{:,6:end},1);
neg_std=std(NegativeControlCountTable{:,6:end},1,1);
pos_e=PositiveControlCountTable{PositiveControlCountTable.Name=="POS_E(0.5)",6:end};
bkg_threshold=neg_mean+2*neg_std;
QCLoD=pos_e>bkg_threshold;
% Threshold endogenous data:
dataForHM=EndogenousCountTable{:,6:end};
for col=1:size(dataForHM,2)
dataForHM(dataForHM(:,col)<bkg_threshold(col))=bkg_threshold(col);
end
%Normalization:
%POS control normalization:
geomeanPOS=geomean(PositiveControlCountTable{1:5,6:end},1);
armeanPOS=mean(geomeanPOS);
POSnorm=armeanPOS./geomeanPOS;
dataForHM=dataForHM.*POSnorm;

%CodeSet control normalization:
geomeanCodeSet=geomean(HousekeepingCountTable{1:5,6:end},1);
armeanCodeSet=mean(geomeanCodeSet);
CodeSetnorm=armeanCodeSet./geomeanCodeSet;
dataForHM=dataForHM.*CodeSetnorm;
dataForHM(dataForHM<1)=1;


%% VENN charts
% totalCropsNum=[153586,107573,301307,279821];

addpath downloadedFromMatCentral/venn/
common_brc=[sum(ismember(L1_brc_list_gt.ImNum,L1_brc_listpred.ImNum)),sum(ismember(L2_brc_list_gt.ImNum,L2_brc_listpred.ImNum)),sum(ismember(L3_brc_list_gt.ImNum,L3_brc_listpred.ImNum)),sum(ismember(L4_brc_list_gt.ImNum,L4_brc_listpred.ImNum))];
identical_brc=[sum(ismember(L1_brc_list_gt,L1_brc_listpred,'rows')),sum(ismember(L2_brc_list_gt,L2_brc_listpred,'rows')),sum(ismember(L3_brc_list_gt,L3_brc_listpred,'rows')),sum(ismember(L4_brc_list_gt,L4_brc_listpred,'rows'))];
% crop_sz=[size(L1gtStack,4),size(L2gtStack,4),size(L3gtStack,4),size(L4gtStack,4)];
identified_sz=[size(L1_brc_list_gt,1), size(L2_brc_list_gt,1),size(L3_brc_list_gt,1),size(L4_brc_list_gt,1)
    size(L1_brc_listpred,1),size(L2_brc_listpred,1),size(L3_brc_listpred,1),size(L4_brc_listpred,1)];



 sampleLabels=["HC1","UC1","UC2","HC2"];

% predlabels={'179','178.5','179','178.5'};
% predlabels={'178','178','178','178'};

xlims=[-300 350];
ylims= [-350 400];
C=colororder('default');
figure(2)
clf
L=[1,4,2,3];
% L=4;
t=tiledlayout(length(L),2,'TileSpacing','compact','Padding','tight');

for l=1:length(L)
nexttile
A_gt_pred=identified_sz(:,L(l));
I_common=common_brc(L(l));
venn(A_gt_pred,I_common,'FaceColor',{ C(3,:),C(4,:)},'FaceAlpha',{0.5,0.5},'EdgeColor','black');
% title('Common out of total reads');
legend({['GT - ',int2str(identified_sz(1,L(l))),' barcodes'],['Pred - ',int2str(identified_sz(2,L(l))),' barcodes']})
text(0,0,['Common barcodes: ',int2str(I_common),10,'Common ratio: ',num2str(I_common/max(A_gt_pred),2)],'FontSize',16,'HorizontalAlignment',  'center');
set(gca,'FontSize',16,'FontWeight','Bold')
h1=gca;
h1.XTick=[];
h1.YTick=[];
h1.Box='off';
xlim(xlims)
ylim(ylims)
ylabel(sampleLabels(L(l)))
nexttile
A_common=repmat(I_common,1,2);
I_identical=identical_brc(L(l));
A_common(2)=I_identical;

[V,S]=venn(A_common,I_identical,'FaceColor',{ C(2,:),C(1,:)},'FaceAlpha',{0.5,.5},'EdgeColor','black');
% title('Identical out of common reads',[' (',num2str(A_common(1)),')']);
% title('Identical / common ratio');%,[' (',num2str(A_common(1)),')']);
set(V(1), 'XData', S.Position(2,1) + get(V(1), 'XData'))

legend({['Common Barcodes: ',num2str(A_common(1))],['Identical Barcodes: ',num2str(I_identical)]})
% legend({'GT',['Pred RPA:',predlabels{L(l)}]})

text(0,0,["Identity ratio:",num2str(I_identical/A_common(1),2)],'FontSize',16,'HorizontalAlignment',  'center');
h2=gca;
h2.XTick=[];
h2.YTick=[];
h2.Box='off';
set(gca,'FontSize',16,'FontWeight','Bold')
xlim(xlims)
ylim(ylims)
linkaxes([h1,h2])
end
%% Error Analysis
gtLists={L1_brc_list_gt,L2_brc_list_gt,L3_brc_list_gt,L4_brc_list_gt};
predLists={L1_brc_listpred,L2_brc_listpred,L3_brc_listpred,L4_brc_listpred};
LaneLabels={'HC1','UC1','UC2','HC2'};
newColorMap=[ 
    0.9691   0         0 
    1.0000    0.8392         0
    0    0.9156    0.8658
         0    .8         0];
figure(4)
t=tiledlayout(4,2,"TileSpacing","compact","Padding","compact");

for i=1:4
    L1_brc_list_gt=gtLists{L(i)};
    L1_brc_listpred=predLists{L(i)};
L1mismatch_gt=L1_brc_list_gt(ismember(L1_brc_list_gt.ImNum,L1_brc_listpred.ImNum)&(~ismember(L1_brc_list_gt,L1_brc_listpred,'rows')),:);
L1mismatch_pred=L1_brc_listpred(ismember(L1_brc_listpred.ImNum,L1_brc_list_gt.ImNum)&(~ismember(L1_brc_listpred,L1_brc_list_gt,'rows')),:);
mismatchreads=L1mismatch_gt.Barcode~=L1mismatch_pred.Barcode;
isGY=sum((L1mismatch_gt.Barcode(:,1:end-1)==2&L1mismatch_gt.Barcode(:,2:end)==4)|(L1mismatch_gt.Barcode(:,1:end-1)==4&L1mismatch_gt.Barcode(:,2:end)==2),2)>0;
singleErrormismatch=sum(mismatchreads,2)<=1;
single_mismatchreads=mismatchreads(singleErrormismatch,:);
errors_gt=L1mismatch_gt.Barcode(singleErrormismatch,:);
errors_gt=errors_gt(single_mismatchreads);

errors_pred=L1mismatch_pred.Barcode(singleErrormismatch,:);
errors_pred=errors_pred(single_mismatchreads);
num_single_mismatch=sum(singleErrormismatch)
num_mismatches=size(mismatchreads,1)

explode=[0 1 0 1];

ax=nexttile;
hist=histogram(errors_gt,4);
gtmismatchHist=hist.Values;
values=compose('%.0f',gtmismatchHist);
percentage=compose(' (%.0f%%)',gtmismatchHist./sum(gtmismatchHist).*100);

hpie=pie(hist.Values,explode,strcat(values,percentage));
set(hpie([2,4,6,8]),'FontSize',14);

set(gca,'ColorMap',newColorMap,'FontSize',20)
if i==1
title(['Ground-truth',newline],'FontSize',16,'FontWeight','bold')
end
ylabel(ax,LaneLabels(i))

nexttile
hist=histogram(errors_pred,4);
predmismatchHist=hist.Values;
values=compose('%.0f',predmismatchHist);
percentage=compose(' (%.0f%%)',predmismatchHist./sum(predmismatchHist).*100);

hpie=pie(hist.Values,explode,strcat(values,percentage));
set(hpie([2,4,6,8]),'FontSize',14);
set(gca,'ColorMap',newColorMap,'FontSize',20)
if i==1
title(['Prediction',newline],'FontSize',16,'FontWeight','bold')
end
end
title(t,"Barcode mismatches - Error distribution",'FontSize',20,'FontWeight','bold  ')
% legend({'Red','Yellow','Blue','Green'},'Location','SouthWestOutside')

%% plot bar graph comparison (counts 2-yaxis)
C=colororder('default');
sampleLabels=["HC1","UC1","UC2","HC2"];

methodLabels=["nCounter","gt","pred RPA:178"
    "nCounter","gt","pred RPA:178"
    "nCounter","gt","pred RPA:178"
    "nCounter","gt","pred RPA:178"];
num_barcodes=25;
f=figure(3);
clf
L=[1,4,2,3];
L=4;

t1=tiledlayout(length(L),1,"TileSpacing","compact");
for l=1:length(L)
Lcols = find(contains(EndogenousCountTable.Properties.VariableNames,string(['L',int2str(L(l))])));
[OrderedcountsPlot,indx]=sortrows(EndogenousCountTable,Lcols(1),"descend");

set(f,'defaultAxesColorOrder',[ C(1,:) ;[0 0 0]]);

nexttile


yyaxis left
colororder('default');
bl=bar((1:num_barcodes)',[OrderedcountsPlot{1:num_barcodes,Lcols(1)},zeros(num_barcodes,2)],'grouped');
% ylim('tight');
% y_liml=ylim;
maxCount=max(OrderedcountsPlot{1:num_barcodes,Lcols(1)},[],'all');

ylim([0,maxCount+maxCount*.05]);
ylabel('nCounter reads')
yyaxis right
%           colororder(C(2:3,:));
colororder('default')

br=bar((1:num_barcodes)',[zeros(num_barcodes,1),OrderedcountsPlot{1:num_barcodes,Lcols(2:3)}],'grouped');
% ylim('tight');
% y_limr=ylim;
maxCount=max(OrderedcountsPlot{1:num_barcodes,Lcols(2:3)},[],'all');
ylim([0,maxCount+maxCount*.05]);
    % bn=bar(sortrows(EndogenousCountTable{1:num_barcodes,Lcols},1,"descend"),'grouped');
xticks(1:num_barcodes)
% oldstr={'1','2','3','4','  '};
% newstr={'R', 'Y', 'B','G',','};
barcode_labels=string(OrderedcountsPlot.Name(1:num_barcodes));
% for chr=1:length(oldstr)
%     barcode_labels=strrep(barcode_labels,oldstr{chr},newstr{chr});
% end
ylabel('CoCoS reads')

% barcode_labels= '['+barcode_labels+']';
xticklabels(barcode_labels)
xtickangle(gca,45)
set(gca,'FontSize',16,'FontWeight','bold','LineWidth',2)
legend([bl,br],methodLabels(L(l),:))
title(sampleLabels(L(l)))

end
xlabel(t1,'Barcodes','FontSize',20)
% ylabel(t,'Barcode counts')
title(t1,['Barcode reads distribution comparison - ',num2str(num_barcodes),' most abundant barcodes'],'FontSize',20)
yyaxis right
% ylabel(t,'Barcode counts CoCoS-DL')

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTIONS:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% full process from crops to barcode counts function
function [barcodeCountTable,gt_list,gt_stack,gt_corrected_image_stack,gt_meanBleedParam]=processCrops(CropImPath,CropImMat, sampleName,meanBleedParam)
if isempty(CropImMat)    
gt_stack=extractTif(CropImPath,4);
else
    gt_stack=CropImMat;
end
    [gt_profile,gt_Xprofile,gtXrange]=CreateProfiles(gt_stack);
    barcode_numbers_list=(1:size(gt_stack,4))';


gt_profile_norm=gt_profile-min(gt_profile(1:size(gt_profile,1),:,:),[],1);
gt_Xprofile_norm=gt_Xprofile-min(gt_Xprofile(1:size(gt_Xprofile,1),:,:),[],1);

warning('off', 'MATLAB:mir_warning_maybe_uninitialized_temporary');
cd( fullfile("downloadedFromMatCentral","FastPsfFitting_26July2016"));
[gt_indForCal,~,gt_Psfs_wo_yellow,~]= imagesForBleedCalc(gt_stack,gt_profile_norm,gtXrange,barcode_numbers_list);
if isempty(meanBleedParam)
[gt_meanBleedParam,~,~]=bleedthroughParamCalc(gt_stack,gt_indForCal,gt_Psfs_wo_yellow);
else
    gt_meanBleedParam=meanBleedParam;
end
gt_corrected_image_stack= CorrectBleedthroughStack(gt_stack,gt_Psfs_wo_yellow,gt_meanBleedParam);

[gt_Psfs,~]=readYellowChannel(gt_corrected_image_stack,gtXrange,gt_Psfs_wo_yellow,barcode_numbers_list);
[gtPsf_bad_brc_ind,gtPsfbarcodeTable,~]=PsfsFilterToBarcodes(gt_Psfs,barcode_numbers_list,0);
gt_list=sortrows(gtPsfbarcodeTable,{'ImNum'});



gt_barcode_cat=categorical(string(num2str(gt_list.Barcode)));
combined_barcodes=categories([gt_barcode_cat]);
gt_counts=histcounts(gt_barcode_cat,combined_barcodes)';
barcodeCountTable=sortrows(table(categorical(combined_barcodes),gt_counts),2,'descend');
barcodeCountTable.Properties.VariableNames{2} = ['counts ',sampleName] ;

end





%% Double peak extraction
function [adj_pks,adj_locs,adj_w,r2]=doublePeakExtraction(profile_ch,n_rec_peaks,locs,pks,w,Min2pksW,brc,ch)
x_L=1;
x_R=length(profile_ch);
douP_ind=find(w>Min2pksW);
if n_rec_peaks==2
    locs_ind(1)=floor(locs(1));
    locs_ind(2)=ceil(locs(2));
    [~,loc_min]=min(profile_ch(locs_ind(1):locs_ind(2)),[],'linear');
    loc_min=loc_min+locs_ind(1)-1;
    if locs(douP_ind)>locs(w<=Min2pksW)
        if ~isempty(find(profile_ch(loc_min:locs_ind(douP_ind))<pks(douP_ind)/2,1,'last'))
            x_L=max(max(loc_min+find(profile_ch(loc_min:locs_ind(douP_ind))<pks(douP_ind)/2,1,'last')-3,loc_min),x_L);
        else
            x_L=max(loc_min,x_L);
        end
        if ~isempty(find(profile_ch(locs_ind(douP_ind):x_R)<pks(douP_ind)/2,1,'first'))
            x_R=min(locs_ind(douP_ind)+find(profile_ch(locs_ind(douP_ind):x_R)<pks(douP_ind)/2,1,'first'),x_R);
        end
    else
        if ~isempty(find(profile_ch(locs_ind(douP_ind):loc_min)<pks(douP_ind)/2,1,'first'))
            x_R=min(min(locs_ind(douP_ind)+find(profile_ch(locs_ind(douP_ind):loc_min)<pks(douP_ind)/2,1,'first'),loc_min),x_R);
        else
            x_R=min(loc_min,x_R);
        end
        if ~isempty(find(profile_ch(x_L:locs_ind(douP_ind))<pks(douP_ind)/2,1,'last'))
            x_L=max(find(profile_ch(x_L:locs_ind(douP_ind))<pks(douP_ind)/2,1,'last')-2,x_L);
        end
    end
elseif  n_rec_peaks==1
    locs_ind=round(locs);
    if ~isempty(find(profile_ch(x_L:locs_ind(douP_ind))<pks(douP_ind)/2,1,'last'))
        x_L=max(find(profile_ch(x_L:locs_ind(douP_ind))<pks(douP_ind)/2,1,'last')-2,x_L);
    end
    if ~isempty(find(profile_ch(locs_ind(douP_ind):x_R)<pks(douP_ind)/2,1,'first'))
        x_R=min(locs_ind(douP_ind)+find(profile_ch(locs_ind(douP_ind):x_R)<pks(douP_ind)/2,1,'first'),x_R);
    end
end
x=floor(x_L):ceil(x_R);
if length(x)<5
    adj_pks=1;
    adj_locs=1;
    adj_w=1;
    r2=0;
    return
elseif length(x)<6
    if min(x)~=1

        x=[min(x)-1,x];
    else
        x=[x,max(x)+1];
    end
end
try
    peak_profile=profile_ch(x);
catch
    warning(['Problem in brc:', num2str(brc),' ch:', num2str(ch),' x:',num2str(x)]);
end
upper_b=ones(1,6);
upper_b([1,4])=max(peak_profile)*0.8;
upper_b([3,6])=3;
upper_b(5)=x_R-1.5;
upper_b(2)=floor(x_L+(x_R-x_L)/2);

lower_b= ones(1,6);
lower_b([1,4])=maxk(peak_profile,1)/3;
lower_b([3,6])=1.5;
lower_b(2)=x_L+1.5;
lower_b(5)=ceil(x_L+(x_R-x_L)/2);

start_p([1,4])=max(peak_profile)/2;
start_p([3,6])=min(w/5,3);
start_p([2,5])=[locs(douP_ind)-w(douP_ind)/4,locs(douP_ind)+w(douP_ind)/4];

gauss_fittype=fittype('gauss2');
options = fitoptions(gauss_fittype);
options.Upper =upper_b;
options.Lower =lower_b;
options.StartPoint =start_p;
[f,gof]=fit(x',peak_profile,gauss_fittype,options);
r2=gof.rsquare;
coeff_values=coeffvalues(f);
adj_locs=[coeff_values([2,5])';locs(w<=Min2pksW)];
adj_pks=[coeff_values([1,4])';pks(w<=Min2pksW)];
adj_w=[coeff_values([3,6])';w(w<=Min2pksW)];

end

%% Single peak gaussian extraction
function [adj_pk,adj_loc,adj_w,r2]=singlePeakExtraction(profile_ch,loc,pk,w,brc,ch)

adj_loc=loc;
adj_pk=pk;
adj_w=w;

x_L=max(floor(loc-w/2-1),1);
x_R=min(ceil(loc+w/2),length(profile_ch));
x=x_L:x_R;

local_min=find(islocalmin(profile_ch(x)))';
if ~isempty(local_min)&&length(local_min)<3
    relPos=local_min./length(x);
    minR_TF=ceil(relPos)==round(relPos);
    if sum(minR_TF)<2
        if ~isempty(local_min(minR_TF))
            x_R=min([x_L+local_min(minR_TF),x_R]);
        elseif ~isempty(local_min(~minR_TF))
            x_L=max([x_L+local_min(~minR_TF),x_L]);
        end
    end
end
x=x_L:x_R;
base=min(profile_ch(x));
peak_profile=profile_ch(x)-base;
pk=pk-base;

upper_b(1)=pk+pk*0.1;
upper_b(2)=x_R-1;
upper_b(3)=w/2;
lower_b(1)=pk/3;
try
    lower_b(2)=x_L+1;
catch
    warning(['x_L:',num2str(x_L),' length:',num2str(length(x_L)),'in brc:', num2str(brc), 'ch:',num2str(ch)]);
    lower_b(2)=floor(loc-1);
end
lower_b(3)=1.5;
start_p(1)=pk;
start_p(3)=min(1.4*(w/2.5),w/2);
start_p(2)=loc;
gauss_fun=@(p,xdata) p(1) .* exp(-((xdata - p(2))/p(3)).^2);


options = optimoptions('lsqcurvefit','Display','off');
[a,resnorm,~,r2,~] = lsqcurvefit(gauss_fun,start_p,x,peak_profile',lower_b,upper_b,options);

if r2>0.9
    adj_loc=a(2);
    adj_pk=a(1)+base;
    adj_w=a(3);
    %     plot(x_fit,gauss_fun(a,x_fit)+ base,'-k')
    %     text(adj_loc,adj_pk,[num2str(resnorm,4),' w:',num2str(adj_w,2)])

end


end

%% Filter barcodes according to processed peaks readout
function [bad_brc,barcodeRead]=barcodeFilter(barcode_mat)
% bad_brc code for errors:
% bad_brc=4:  LocsDiff<minDistBetweenAdjLocs
% bad_brc=5:  LocsDiff>maxDistBetweenAdjLocs
% bad_brc=6:  total size of barcode is not 6
% bad_brc=7:  more than 3 spots per channel
% bad_brc=8:  adjacent same colors

bad_brc=0;
barcodeRead=zeros(6,1);
barcode_output=barcode_mat;
pksBarcode=barcode_mat(:,1);
locsBarcode=barcode_mat(:,2);
chBarcode=barcode_mat(:,3);

FilterOptions.maxLocsperCh=3;
FilterOptions.minDistBetweenAdjLocs=.5;
FilterOptions.maxDistBetweenAdjLocs=5;
FilterOptions.TotalPks=6;
FilterOptions.YGRatio=1;
FilterOptions.minDistBetweenYG=2;
FilterOptions.minTotalDist=8;

LocsDiff=diff(locsBarcode);
YGcheckInd=(chBarcode(1:end-1)==2&chBarcode(2:end)==4)&(LocsDiff<FilterOptions.minDistBetweenYG);
GYcheckInd=(chBarcode(1:end-1)==4&chBarcode(2:end)==2)&(LocsDiff<FilterOptions.minDistBetweenYG);
indToRemove=[];
if size(barcode_output,1)>FilterOptions.TotalPks
    % Looking for overlapping green and yellow, and throwing away the yellows.
    for  yg_ind=find(YGcheckInd)'
        if pksBarcode(yg_ind)/pksBarcode(yg_ind+1)<FilterOptions.YGRatio
            indToRemove=[indToRemove,yg_ind];
        end
    end
    for  gy_ind=find(GYcheckInd)'
        if pksBarcode(gy_ind+1)/pksBarcode(gy_ind)<FilterOptions.YGRatio
            indToRemove=[indToRemove,gy_ind+1];
        end
    end
    barcode_output(indToRemove,:)=[];
    chBarcode(indToRemove)=[];
    locsBarcode(indToRemove)=[];
    LocsDiff=diff(locsBarcode);
end

%Looking for too close peaks after YG combining
if sum(LocsDiff<FilterOptions.minDistBetweenAdjLocs)
    for diff_ind=find(LocsDiff<FilterOptions.minDistBetweenAdjLocs)'
        if ~(chBarcode(diff_ind)==2&&chBarcode(diff_ind+1)==4)&&~(chBarcode(diff_ind)==4&&chBarcode(diff_ind+1)==2)
            bad_brc=4;
            return
        end
    end
end


%Looking for too far apart peaks:
if sum(LocsDiff>FilterOptions.maxDistBetweenAdjLocs)
    bad_brc=5;
    return
end
%Making sure only 6 points per barcode read
if size(barcode_output,1)~=FilterOptions.TotalPks
    bad_brc=6;
    return

end
%Making sure only 3 points per channels
for ch=1:4
    if sum(chBarcode==ch)>FilterOptions.maxLocsperCh
        bad_brc=7;
        return

    end
end



%No adjacent same color points
if sum(diff(chBarcode)==0)>0
    bad_brc=8;
    return
end

%Total length of the barcode is higher than 8 pixels
if ( locsBarcode(end)-locsBarcode(1))<FilterOptions.minTotalDist
    bad_brc=9;
    return
end

if ~bad_brc
    barcodeRead=barcode_output(:,3);
end

end
%% Psfs filter to readout barcodes
function [bad_brc_ind,barcodeTable,BarcodeCell]=PsfsFilterToBarcodes(Psfs_cell,barcode_numbers_list,profile_flag)
%Input Psfs rows:
% 1- x location 
% 2- y location 
% 3- amplitude 
% 4- bkg 
% 5- sigma 
% 6-8 irrelevant
% 9- channel
%
% bad_brc code for errors: 
% bad_brc=4:  LocsDiff<minDistBetweenAdjLocs
% bad_brc=5:  LocsDiff>maxDistBetweenAdjLocs 
% bad_brc=6:  total spots of barcode is not 6 
% bad_brc=7:  more than 3 spots per channel 
% bad_brc=8:  adjacent same colors 
% bad_brc=9:  total length is less than 8 pix


Filter.maxLocsperCh=3;
Filter.minDistBetweenAdjLocs=.5;
Filter.maxDistBetweenAdjLocs=5;
Filter.TotalPks=6;
Filter.minDistBetweenYG=1;
Filter.minTotalDist=6;
Filter.minDistToUniteReads=1;
Filter.minIyellow=150;
Filter.MinWYellow=0.7;
Filter.MaxXmismatch=2;

% if profile_flag
%     Filter.minIyellow=150/0.34; %normalizing the intensity from 2d to 1d sum of 2d gaussian by multiplying the faction of 1/34%
% 
% end


wb=waitbar(0,'Filtering Psfs and reading barcodes');
D = parallel.pool.DataQueue;
afterEach(D,@parforWaitbar);

bad_brc_ind=zeros(1,numel(Psfs_cell));
six_or_more_psfs=cellfun(@(x) size(x,2)>=Filter.TotalPks,Psfs_cell,'UniformOutput',true);
lessthan6=cellfun(@(x) size(x,2)<Filter.TotalPks,Psfs_cell,'UniformOutput',true);
bad_brc_ind(lessthan6)=6;

Psfs_to_check=Psfs_cell(six_or_more_psfs);
% morethan6=cellfun(@(x) size(x,2)>Filter.TotalPks,Psfs_to_check,'UniformOutput',true);
N=numel(Psfs_to_check);
bad_brc=zeros(1,N);
parforWaitbar(wb,N);
parfor brc=1:N
    b=array2table(Psfs_to_check{brc}',"VariableNames",{'xlocs','ylocs','I','bkg','w','irr1','irr2','irr3','ch'});%([1:5,9],:)

    if size(b,1)>Filter.TotalPks
  
    
% Remove yellows overlapping with green marks
    checkAdjYG=(b.ch(1:end-1)==2&b.ch(2:end)==4)|(b.ch(1:end-1)==4&b.ch(2:end)==2);
    checkAdjDistYG=diff(b.ylocs)<Filter.minDistBetweenYG&checkAdjYG;
        if sum(checkAdjDistYG)
            checkAdjDistYG(numel(checkAdjDistYG)+1)=false;
            checkAdjDistYG(find(checkAdjDistYG)+1)=true;
            b(b.ch==2&checkAdjDistYG,:)=[];
        end

% Remove yellows with Intensities too low (residuals from bleedthrough
% correction)
        
    checkMinIyellow=b.I<Filter.minIyellow & b.ch==2&b.w<Filter.MinWYellow;
        if sum(checkMinIyellow)&&size(b,1)>Filter.TotalPks
            b(checkMinIyellow,:)=[];
        end
% Remove duplicate calls of the same color within the same Psf and keep the
% higher Intensity call.
    checkAdjSameColor = (diff(b.ch)==0);
    checkAdjDistUnite = diff(b.ylocs)<Filter.minDistToUniteReads;
        if sum(checkAdjSameColor)
            if sum(checkAdjSameColor & checkAdjDistUnite)
              ind= find(checkAdjSameColor & checkAdjDistUnite);
              b(ind,:)=b(ismember(b.I,max(b.I(ind),b.I(ind+1))),:);
              b(ind+1,:)=[];
              
             

            else
            bad_brc(brc)=8;
            end
        end

% check that adjacent points are separated enough, if not than check that
% it isn't due to a false detection outside of the barcode

    checkMinAdjDist=diff(b.ylocs)<Filter.minDistBetweenAdjLocs;
   
    if sum(checkMinAdjDist)
        checkMinAdjDist(numel(checkMinAdjDist)+1)=false;
        checkMinAdjDist(find(checkMinAdjDist)+1)=true;
        X=[ones(length(b.xlocs),1) b.ylocs];
        Y=b.xlocs;
        B=X\Y;
        checkMismatchX=abs(b.xlocs-X*B)>Filter.MaxXmismatch;
        if sum(checkMismatchX&checkMinAdjDist)
            b(checkMismatchX&checkMinAdjDist,:)=[];
            checkMinAdjDist=diff(b.ylocs)<Filter.minDistBetweenAdjLocs;
            if sum(checkMinAdjDist)
                bad_brc(brc)=4;
            end
        else
            bad_brc(brc)=4;
        end
    end


% Check number of calls of the same color is less than 3
    checkMaxPointPerCh=histcounts(b.ch,1:5)>Filter.maxLocsperCh;
    if sum(checkMaxPointPerCh)
        bad_brc(brc)=7;
    end
    
% check that this is not a short bulb/
    if (b.ylocs(end)-b.ylocs(1))<Filter.minTotalDist
        bad_brc(brc)=9;
    end

    if size(b,1)~=Filter.TotalPks
        bad_brc(brc)=6;
    elseif sum(diff(b.ch)==0)
        bad_brc(brc)=8;
    end
    
    Psfs_to_check{brc}=table2array(b)';



    else%if there are exactly 6 points, check that everything in order
        %without removing or merging points

   % check for adjacent points with same color      
    if sum(diff(b.ch)==0)
        bad_brc(brc)=8;
    end
    % Check number of calls of the same color is less than 3
    checkMaxPointPerCh=histcounts(b.ch,1:5)>Filter.maxLocsperCh;
    if sum(checkMaxPointPerCh)
        bad_brc(brc)=7;
    end
% check that adjacent points are separated enough 
    checkMinAdjDist=diff(b.ylocs)<Filter.minDistBetweenAdjLocs;
    if sum(checkMinAdjDist)
         bad_brc(brc)=4;
    end
    % check that this is not a short bulb/
    if (b.ylocs(end)-b.ylocs(1))<Filter.minTotalDist
        bad_brc(brc)=9;
    end
    end
send(D,[]);
end
delete(wb);
bad_brc_ind(six_or_more_psfs)=bad_brc;
Psfs_cell(six_or_more_psfs)=Psfs_to_check;
BarcodeCell=Psfs_cell(~bad_brc_ind);
barcodeTable=cell2table(cellfun(@(x) x(9,:),BarcodeCell,'UniformOutput',false)','VariableNames',{'Barcode'});
barcodeTable.ImNum=barcode_numbers_list(~bad_brc_ind);
% barcodeTable=table(BarcodeReadout,...
%     'VariableNames',{'Barcode'},...
%     'RowNames',string(barcode_numbers_list(~bad_brc_ind)));



end
%% Plot profiles
function plotProfiles(profiles,barcode_numbers,fig_num)
ProminenceToPeakFit=200;
MinWidthFor2PeaksAnalysis=6.5;
WidthForSinglePeakFit=3;
MinPeakHeights=[1000,500,500,1000];
figure(fig_num)
clf
tiledlayout('flow','TileSpacing', 'compact','Padding','compact')
C=colororder([ 0.8 0 0
    0.8 0.6 0
    0.2 0.8 0.8
    0 0.8 0]);

for brc=barcode_numbers
    nexttile
    barcode_mat=[];
    bad_brc=0;
    adj_profile=squeeze(profiles(:,:,brc));
    %     adj_profile(:,2)=adj_profile(:,2)-adj_profile(:,4);
    adj_profile(:,2)=adj_profile(:,2)-(adj_profile(:,4)./max(adj_profile(:,4)).*max(adj_profile(:,2)));
    %     adj_profile(:,2)=adj_profile(:,2)-max(adj_profile(:,2)).*(adj_profile(:,2)./max(adj_profile(:,2))).*(adj_profile(:,4)./max(adj_profile(:,4)));
    adj_profile((adj_profile(:,2)<0),2)=0;
    adj_profile(:,2)=adj_profile(:,2)+adj_profile(:,4).*(adj_profile(:,2)./max(adj_profile(:,2))).*(adj_profile(:,4)./max(adj_profile(:,4)));

    hold on
    for ch=1:size(adj_profile,2)
        findpeaks(adj_profile(:,ch),'NPeaks',3, 'MinPeakProminence',100,'MinPeakHeight',MinPeakHeights(ch),'Annotate','extents','WidthReference','halfheight');

        [pks,locs,w,p] =findpeaks(adj_profile(:,ch),'NPeaks',3, 'MinPeakProminence',100,'MinPeakHeight',MinPeakHeights(ch),'Annotate','extents','WidthReference','halfheight');
        n_rec_peaks=length(w);
        for pks_ind=find(p>ProminenceToPeakFit&w<MinWidthFor2PeaksAnalysis&w>WidthForSinglePeakFit)'
            [fit_pk,fit_loc,fit_w,r2]=singlePeakExtraction(adj_profile(:,ch),locs(pks_ind),pks(pks_ind),w(pks_ind),brc,ch);
            if r2>0.9
                pks(pks_ind)=fit_pk;
                locs(pks_ind)=fit_loc;
                w(pks_ind)=fit_w;
            end
        end


        if ~isempty(w(w>MinWidthFor2PeaksAnalysis))
            if sum(w>MinWidthFor2PeaksAnalysis)>1
                bad_brc=1;
                break;
            end

            if n_rec_peaks>2
                bad_brc=2;
                break;
            else
                [fit_pks,fit_locs,fit_w,r2]=doublePeakExtraction(adj_profile(:,ch),n_rec_peaks,locs,pks,w,MinWidthFor2PeaksAnalysis,brc,ch);

                if r2<0.9
                    if w(w>MinWidthFor2PeaksAnalysis)>MinWidthFor2PeaksAnalysis+2
                        bad_brc=3;
                        break;
                    else
                        [fit_pk,fit_loc,fit_w,r2]=singlePeakExtraction(adj_profile(:,ch),locs(w>MinWidthFor2PeaksAnalysis),pks(w>MinWidthFor2PeaksAnalysis),w(w>MinWidthFor2PeaksAnalysis),brc,ch);
                        if r2>0.9
                            pks(w>MinWidthFor2PeaksAnalysis)=fit_pk;
                            locs(w>MinWidthFor2PeaksAnalysis)=fit_loc;
                            w(w>MinWidthFor2PeaksAnalysis)=fit_w;
                        end
                    end
                else
                    pks=fit_pks;
                    locs=fit_locs;
                    w=fit_w;
                end
            end

        end
        %         [locs,sorted_ind]=sort(locs,'ascend');
        %         pks=pks(sorted_ind);
        %         w=w(sorted_ind);
        barcode_mat=vertcat(barcode_mat,[pks,locs,repmat(ch,length(locs),1)]);
        for xline_ind=1:length(locs)
            xline(locs(xline_ind),'linewidth',2,'Color',C(ch,:));
        end
    end
    if ~bad_brc
        [bad_brc2,barcodeRead]=barcodeFilter(sortrows(barcode_mat,2));
    else
        barcodeRead=zeros(6,1);
        %         if ~bad_brc2
        % %             barcode_array(brc,:)=barcodeRead';
        %
        %         end

    end


    %     %%
    %     barcode_mat=[];
    %     bad_brc=false;
    %     adj_profile=squeeze(profiles(:,:,brc));
    %     adj_profile(:,2)=adj_profile(:,2)-adj_profile(:,4);
    %     hold on
    %     for ch=1:size(adj_profile,2)
    %         findpeaks(adj_profile(:,ch),'NPeaks',3, 'MinPeakProminence',100,'MinPeakHeight',700,'Annotate','extents','WidthReference','halfheight');
    %         [pks,locs,w,p] =findpeaks(adj_profile(:,ch),'NPeaks',3, 'MinPeakProminence',100,'MinPeakHeight',700,'Annotate','extents','WidthReference','halfheight');
    %         for pks_ind=find(p>ProminenceToPeakFit&w<MinWidthFor2PeaksAnalysis&w>3)'
    %             [adj_pk,adj_loc,adj_w,r2]=singlePeakExtraction(adj_profile(:,ch),locs(pks_ind),pks(pks_ind),w(pks_ind),brc,ch);
    %             if r2>0.9
    %                 pks(pks_ind)=adj_pk;
    %                 locs(pks_ind)=adj_loc;
    %                 w(pks_ind)=adj_w;
    %             end
    %         end
    %
    %
    %         if ~isempty(w(w>MinWidthFor2PeaksAnalysis))
    %             if sum(w>MinWidthFor2PeaksAnalysis)>1
    %                 bad_brc=1;
    %                 barcode_mat=zeros(6,3);
    %                 break;
    %             end
    %             n_rec_peaks=length(w);
    %             if n_rec_peaks>2
    %                 bad_brc=2;
    %                 barcode_mat=zeros(6,3);
    %
    %                 break;
    %             else
    %                 [pks,locs,w,r2]=doublePeakExtraction(adj_profile(:,ch),n_rec_peaks,locs,pks,w,MinWidthFor2PeaksAnalysis,brc,ch);
    %
    %                 if r2<0.9
    %                     bad_brc=3;
    %                     barcode_mat=zeros(6,3);
    %
    %                     break;
    %                 end
    %             end
    %
    %         end
    %     barcode_mat=vertcat(barcode_mat,[pks,locs,repmat(ch,length(locs),1)]);
    %



    %         [bad_brc2,barcodeRead]=barcodeFilter(sortrows(barcode_mat,2));
    title(['Barcode ',num2str(brc),'Read: ',mat2str(barcodeRead'),10, 'flag for bad barcode=',num2str([bad_brc2,bad_brc])])
    sig_h = findobj(gca, 'tag', 'Signal');
    set(sig_h, 'Linewidth',2);
    legend off
    hold off
end

% for brc=1:num_plots
%
%     bad_brc=false;
%
%     nexttile
%     adj_profile=squeeze(profiles(:,:,brc));
%     adj_profile(:,2)=adj_profile(:,2)-adj_profile(:,4);
%
%     for ch=1:size(adj_profile,2)
%         %    adj_profile(:,ch)=smooth(adj_profile(:,ch),'loess');
%
%         hold on
%         findpeaks(adj_profile(:,ch),'NPeaks',3, 'MinPeakProminence',100,'MinPeakHeight',600,'Annotate','extents','WidthReference','halfheight');%[pks,locs,w,p] =
%         [pks,locs,w,p] =findpeaks(adj_profile(:,ch),'NPeaks',3, 'MinPeakProminence',100,'MinPeakHeight',700,'Annotate','extents','WidthReference','halfheight');%[pks,locs,w,p] =
%         legend('off')
%         if ~isempty(w(w>6))
%             x_L=1;
%             x_R=length(adj_profile(:,ch));
%             if length(w)==2
%                 [~,loc_min]=min(adj_profile(locs(1):locs(2),ch),[],'linear');
%                 loc_min=loc_min+locs(1)-1;
%                 if locs(w>6)>locs(w<6)
%                     x_L=max(max(loc_min+find(adj_profile(loc_min:locs(w>6),ch)<pks(w>6)/2,1,'last')-3,loc_min),x_L);
%                     %                     x_L=max(max(floor(locs(w>6)-w(w>6)/1.5-1),loc_min),x_L);
%                     %                     x_R=min(ceil(locs(w>6)+w(w>6)/1.5+1),x_R);
%                     x_R=min(locs(w>6)+find(adj_profile(locs(w>6):x_R,ch)<pks(w>6)/2,1,'first'),x_R);
%                 else
%
%                     x_R=min(min(locs(w>6)+find(adj_profile(locs(w>6):loc_min,ch)<pks(w>6)/2,1,'first'),loc_min),x_R);
%                     %                     x_R=min(min(ceil(locs(w>6)+w(w>6)/1.5+1),loc_min),x_R);
%                     x_L=max(find(adj_profile(x_L:locs(w>6),ch)<pks(w>6)/2,1,'last')-2,x_L);
%                     %                     x_L=max(floor(locs(w>6)-w(w>6)/1.5-1),x_L);
%                 end
%             elseif  length(w)==1
%                 x_L=max(find(adj_profile(x_L:locs(w>6),ch)<pks(w>6)/2,1,'last')-2,x_L);
%                 x_R=min(locs(w>6)+find(adj_profile(locs(w>6):x_R,ch)<pks(w>6)/2,1,'first'),x_R);
%                 %                 max(floor(locs(w>6)-w(w>6)/1.5-1),x_L);
%                 %                 x_R=min(ceil(locs(w>6)+w(w>6)/1.5+1),x_R);
%             else
%                 bad_brc=true;
%                 continue;
%             end
%             x=x_L:x_R;
%             peak_profile=adj_profile(x,ch);%-min(profiles(x,i,j));
%             % norm_peak_profile=peak_profile./max(peak_profile);
%             upper_b=ones(1,6); upper_b([1,4])=max(peak_profile); upper_b([3,6])=3.5; upper_b([2,5])=x_R-3;
%             lower_b= ones(1,6);lower_b([1,4])=max(peak_profile)/3; lower_b([3,6])=2;  lower_b([2,5])=x_L+3;
%             %     fitgmdist(peak_profile,2,'Options',options,'CovarianceType','diagonal');
%             gauss_fittype=fittype('gauss2');
%             options = fitoptions(gauss_fittype);
%             options.Upper =upper_b;
%             options.Lower =lower_b;
%             [f,gof]=fit(x',peak_profile,gauss_fittype,options);
%             r2=gof.rsquare
%             %             plot(f,x',peak_profile,'-ko');
%             if r2<0.9
%                 bad_brc=true;
%                 continue;
%             end
%             coeff_values=coeffvalues(f);
%             adj_locs=coeff_values([2,5])';
%             xline(adj_locs(1),'linewidth',1);
%             xline(adj_locs(2),'linewidth',1);
%
%
%         end
%
%         hold off
%
%
%
%     end
%     sig_h = findobj(gca, 'tag', 'Signal');
%     %     peak_h = findobj(gca, 'tag', 'Peak');
%     %     prom_h = findobj(gca, 'tag', 'Prominence');
%     %     hp_h = findobj(gca, 'tag', 'HalfProminenceWidth');
%     %     line_h = [sig_h, peak_h, prom_h, hp_h];
%     set(sig_h, 'Linewidth',2);
% end
end

%% Extract stacks from Tif files
function B=extractTif(path,num_ch)
num_channels=num_ch;
info = imfinfo(path);
image=imread(path,1);
num_images = numel(info);
% num_images = numel(imfinfo(path));
num_z=num_images/num_channels;
A=zeros(size(image,1),size(image,2),num_images,'uint16');
wb=waitbar(0,'Processing Tiff stack...');
D = parallel.pool.DataQueue;
afterEach(D,@parforWaitbar);
parforWaitbar(wb,num_images)
delete(gcp('nocreate'))
parpool(9)
Cinfo = parallel.pool.Constant(info);
parfor k = 1:num_images
    
%     waitbar(k/num_images,wb);

    Im = imread(path, k, 'Info', Cinfo.Value);
    A(:,:,k)=Im;
    send(D,[]);

end
 B=reshape(A,size(image,1),size(image,2),num_ch,num_z);   

% A(:,:,ch,z)=Im;
delete(wb);
delete(gcp('nocreate'))
[filepath,filename,~]=fileparts(path);
save(fullfile(filepath,[filename,'_stack.mat']),'B','-v7.3','-nocompression') 

end


%% Create Y and X profiles from barcode Tiff stacks
function [Yprofiles,Xprofiles,Xrange]=CreateProfiles(barcodeStack)
Yprofiles=zeros(size(barcodeStack,1),size(barcodeStack,3),size(barcodeStack,4));
wb=waitbar(0,'Creating X and Y profiles');
D = parallel.pool.DataQueue;
afterEach(D,@parforWaitbar);
N=size(barcodeStack,4);
parforWaitbar(wb,N)
grayStack=squeeze(sum(barcodeStack,3));
Xprofiles=squeeze(sum(grayStack,1));
normXprofiles=Xprofiles-min(Xprofiles,[],1);
% smoothNormXprofiles=smoothdata(normXprofiles,1,"gaussian",2);
Xrange=zeros(N,2);
Xlength=size(normXprofiles,1);
parfor brc=1:N
    %     waitbar(brc/size(barcodeStack,4),wb)
    [~,Xlocs,Xw,~]=findpeaks(normXprofiles(:,brc),"NPeaks",1,"SortStr","ascend","WidthReference","halfheight");
%     findpeaks(normXprofiles(:,brc),"NPeaks",1,"SortStr","ascend","WidthReference","halfheight","Annotate","extents");
    if isempty(Xlocs)
        Xlocs=ceil(Xlength/2);
        Xw=Xlength/2;
    end
    lb=max(Xlocs-floor(Xw),1);
    rb=min(Xlocs+floor(Xw),Xlength);
    Yprofiles(:,:,brc)=sum(barcodeStack(:,lb:rb,:,brc),2);
    Xrange(brc,:)=[lb,rb];
    send(D,[]);
end
delete(wb);

end
%% Create Y profiles from barcode Tiff stacks
function Yprofiles=CreateYProfiles(barcodeStack,Xrange)
Yprofiles=zeros(size(barcodeStack,1),size(barcodeStack,3),size(barcodeStack,4));
wb=waitbar(0,'Creating Y profiles');
D = parallel.pool.DataQueue;
afterEach(D,@parforWaitbar);
N=size(barcodeStack,4);
parforWaitbar(wb,N)
parfor brc=1:N
    %     waitbar(brc/size(barcodeStack,4),wb)
    Yprofiles(:,:,brc)=sum(barcodeStack(:,Xrange(brc,1):Xrange(brc,2),:,brc),2);
    send(D,[]);
end
delete(wb);

end

%% parforwaitbar

function parforWaitbar(waitbarHandle,iterations)
persistent count h N

if nargin == 2
    % Initialize

    count = 0;
    h = waitbarHandle;
    N = iterations;
else
    % Update the waitbar

    % Check whether the handle is a reference to a deleted object
    if isvalid(h)
        count = count + 1;
        waitbar(count / N,h);
    end
end
end

%% Finding images without yellow
function [indForCal,imForCal,Psfs,Error_counter]= imagesForBleedCalc(image_stack,yprofile,Xrange,barcode_numbers_list)
MinWidthFor2PeaksAnalysis=6.;
wb=waitbar(0,'Finding images without yellow');
D = parallel.pool.DataQueue;
afterEach(D,@parforWaitbar);
N=size(yprofile,3);
parforWaitbar(wb,3*N)
Psfs=cell(1,N);
channels=[1,3,4];
filter_param_per_ch=[500,100,100,1
    300,50,100,0.5
    500,100,100,1];% minPeakHeight,minPeakProm, minPsfA,minPsfSNR
im_size=size(image_stack,1,2);
Error_counter=false(3,N);
for ch=1:length(channels)
    yprofile_ch=yprofile(:,channels(ch),:);
%     xprofile_ch=xprofile(:,channels(ch),:);
    image_stack_ch=image_stack(:,:,channels(ch),:);
    ch_filter_param=filter_param_per_ch(ch,:);

  parfor profile=1:N
        if max(yprofile_ch(3:end-2,profile))>=ch_filter_param(1)
%             barcode_numbers_list(profile)
            warning('off','signal:findpeaks:largeMinPeakHeight')

            [pks,Ylocs,w,p]=findpeaks(yprofile_ch(:,profile),"NPeaks",3,"SortStr","descend","MinPeakHeight",ch_filter_param(1),"MinPeakProminence",ch_filter_param(2),"MinPeakDistance",2.5,"MinPeakWidth",1.5,"WidthReference","halfheight");
%             findpeaks(yprofile_ch(:,profile),"NPeaks",3,"SortStr","descend","MinPeakHeight",ch_filter_param(1),"MinPeakProminence",ch_filter_param(2),"MinPeakDistance",2.5,"MinPeakWidth",1.5,"WidthReference","halfheight",'Annotate','extents');
            warning('on','signal:findpeaks:largeMinPeakHeight')
            Xlocs=mean(Xrange(profile,:));
%             [~,Xlocs]=findpeaks(xprofile_ch(:,profile),"NPeaks",1,"MinPeakHeight",ch_filter_param(1),"MinPeakProminence",ch_filter_param(2),"MinPeakDistance",3);
        else
            Ylocs=[];
            Xlocs=[];
        end
        if ~isempty(Ylocs)&&~isempty(Xlocs)
            n_rec_peaks=length(w);
            if sum(w>MinWidthFor2PeaksAnalysis)&&n_rec_peaks<3
                if sum(w>MinWidthFor2PeaksAnalysis)==2
                    w(w~=max(w))=MinWidthFor2PeaksAnalysis-.1;
                end

                %                 barcode_numbers_list(profile)

                try
                    [fit_pks,fit_locs,fit_w,r2]=doublePeakExtraction(yprofile_ch(:,profile),n_rec_peaks,Ylocs,pks,w,MinWidthFor2PeaksAnalysis,barcode_numbers_list(profile),ch);
                catch ME
                    warning('off','backtrace')
                    warning(ME.identifier,'%s problem with double gauss fitting in barcode %d channel %d',ME.message,barcode_numbers_list(profile), channels(ch));
                    warning('on','backtrace')
                    Error_counter(ch,profile)=true;
                    r2=0;
                end
                if r2>0.85
                    Ylocs=fit_locs;
                else

                end

            end
            psfCh=[psfFit_Image(image_stack_ch(:,:,profile),[repmat(Xlocs,1,length(Ylocs));Ylocs'],[],true,[],2);repmat(channels(ch),1,length(Ylocs))];
            if sum( psfCh(5,:)>4|psfCh(5,:)<0.5)
                psfCh(:, psfCh(5,:)>3|psfCh(5,:)<0.5)=[];
            end
            if sum( psfCh(4,:)==0)
                ind0=find(psfCh(4,:)==0);
                for i=1:length(ind0)
                    xpix=round(psfCh(1,ind0(i)));
                    ypix=round(psfCh(2,ind0(i)));
                    lb=max(xpix-3,1);
                    rb=min(xpix+3,im_size(2));
                    ub=min(ypix+3,im_size(1));
                    db=max(ypix-3,1);
                    psfCh(4,ind0(i))= mean([image_stack_ch(ub,lb:rb,profile),image_stack_ch(db,lb:rb,profile),...
                        image_stack_ch(db:ub,lb,profile)',image_stack_ch(db:ub,rb,profile)']);
                end
            end
            if sum( psfCh(3,:)<ch_filter_param(3))
                psfCh(:, psfCh(3,:)<ch_filter_param(3))=[];
            end
            if sum(psfCh(3,:)./psfCh(4,:)<ch_filter_param(4))
                psfCh(:,psfCh(3,:)./psfCh(4,:)<ch_filter_param(4))=[];
            end
            if sum(psfCh(1,:)<2.5|psfCh(1,:)>(im_size(2)-1.5))
                psfCh(:,psfCh(1,:)<2.5|psfCh(1,:)>(im_size(2)-1.5))=[];
            end
            if sum(psfCh(2,:)<2.5|psfCh(2,:)>(im_size(1)-1.5))
            psfCh(:,psfCh(2,:)<2.5|psfCh(2,:)>(im_size(1)-1.5))=[];
            end
            if length(psfCh(2,:))>1
                if sum(diff(sort(psfCh(2,:),'ascend'))<2)
                    psfCh=sortrows(psfCh',2)';
                    while(sum(diff(sort(psfCh(2,:),'ascend'))<=2))
                        ind_minDist=find(diff(psfCh(2,:))==min(diff(psfCh(2,:))));
                        psfCh(:,ind_minDist)=mean(psfCh(:,ind_minDist:ind_minDist+1),2);
                        psfCh(:,ind_minDist+1)=[];
                    end
                end
            end


            Psfs{profile}=[Psfs{profile} psfCh];
        end
        send(D,[]);
    end
end

delete(wb);
% numdet=cellfun(@(x) size(x,2),Psfs,'UniformOutput',true);
% ind_6=find(numdet==6);
% imForCal=barcode_numbers_list(numdet==6);
ind_empty=cellfun(@(x) isempty(x),Psfs,'UniformOutput',true);
Psfs(ind_empty)={zeros(9,1)};
indForCal=find(cellfun(@(x) size(x,2)==6&sum(x(9,:)==4)>0,Psfs,'UniformOutput',true));
imForCal=barcode_numbers_list(indForCal);
Error_counter=sum(Error_counter,"all");
end
%% Matrix bleedthrough correction

function [meanBleedParam,numCalPsfs,numCalIm]=bleedthroughParamCalc(image_stack,indForCal,Psfs_wo_yellow)
wb=waitbar(0,'Finding bleedthrough parameters');
D = parallel.pool.DataQueue;
afterEach(D,@parforWaitbar);
N=length(indForCal);
parforWaitbar(wb,N)
gt_stackCalYellow=double(squeeze(image_stack(:,:,2,indForCal)));
greenOnlyPsfs=cellfun(@(x) x(:,x(9,:)==4),Psfs_wo_yellow(indForCal),'UniformOutput',false);
bleedthroughPsfs=cell(size(greenOnlyPsfs));
im_size=size(image_stack,1,2);
ind_to_del=false(1,N);

parfor im=1:N
    bleedthroughPsfs{im}=psfFit_Image(gt_stackCalYellow(:,:,im),greenOnlyPsfs{im}(1:2,:),[],true,[],2);
    if sum( bleedthroughPsfs{im}(5,:)>2|bleedthroughPsfs{im}(5,:)<0.5)
        ind_to_del(im)=true;
    end
    if sum( bleedthroughPsfs{im}(4,:)<20)
        ind_to_del(im)=true;
    end
    if sum( bleedthroughPsfs{im}(3,:)<100)
        ind_to_del(im)=true;
    end
    if sum(bleedthroughPsfs{im}(1,:)<3|bleedthroughPsfs{im}(1,:)>(im_size(2)-2))
        ind_to_del(im)=true;

    end
    send(D,[]);
end
greenOnlyPsfs(ind_to_del)=[];
bleedthroughPsfs(ind_to_del)=[];
numCalIm=length(bleedthroughPsfs);
delete(wb);
pG=cell2mat(greenOnlyPsfs);
pY=cell2mat(bleedthroughPsfs);
pY(9,:)=2;
numCalPsfs=length(pG);
Delta=[pG(1:2,:)-pY(1:2,:);pG(3:5,:)./pY(3:5,:)];
meanBleedParam=mean(Delta(:,sum(isfinite(Delta),1)==5),2);
end

%% Correct bleedthrough to entire stack
function corrected_image_stack= CorrectBleedthroughStack(image_stack,Psfs_wo_yellow,meanBleedParam)
GreenPsfs=cellfun(@(x) x(:,x(9,:)==4),Psfs_wo_yellow,'UniformOutput',false);

wb=waitbar(0,'Correcting bleedthrogh to yellow channel');
D = parallel.pool.DataQueue;
afterEach(D,@parforWaitbar);
N=size(image_stack,4);
parforWaitbar(wb,N)
% GreenPsfs=cell(1,N);
gaussfun=@(x,y,a) a(3)*exp(-((x-a(1)).^2+(y-a(2)).^2)/(2*a(5)^2));
im_sz=size(image_stack,1,2);
[Xpos,Ypos]=meshgrid(1:im_sz(2),1:im_sz(1));
corrected_image_stack=double(image_stack);
parfor im=1:N

    sim_im=zeros(im_sz(1),im_sz(2));
    simParamY=[GreenPsfs{im}(1:2,:)-meanBleedParam(1:2);GreenPsfs{im}(3:5,:)./meanBleedParam(3:5)];
    for param=1:size(simParamY,2)
        sim_im=sim_im+gaussfun(Xpos,Ypos,simParamY(:,param));
    end
    corrected_image_stack(:,:,2,im)=corrected_image_stack(:,:,2,im)-sim_im;


    send(D,[]);
end
delete(wb);
end
%% Complete barcode readout with corrected yellow channel
function [Psfs,Error_counter]=readYellowChannel(bleedthrough_corr_image_stack,Xrange,Psfs_wo_yellow,barcode_numbers_list)
MinWidthFor2PeaksAnalysis=6.;
minYellowPeakDist=1;
image_stack_Yellow=squeeze(bleedthrough_corr_image_stack(:,:,2,:));
% xprofileYellow=squeeze(CreateXprofiles(image_stack_Yellow));
% [~,xpeakYellow]=max(xprofileYellow(3:end-2,:),[],1);
% Xrange=[xpeakYellow'-2,xpeakYellow'+2]+2;
yprofileYellow=squeeze(CreateYProfiles(image_stack_Yellow,Xrange));
yprofileYellow(yprofileYellow<0)=0;
yprofileYellow_norm=yprofileYellow-min(yprofileYellow,[],1);
wb=waitbar(0,'Reading yellow channel psfs');
D = parallel.pool.DataQueue;
afterEach(D,@parforWaitbar);
N=size(yprofileYellow_norm,2);
parforWaitbar(wb,N)
im_size=size(bleedthrough_corr_image_stack,1,2);
ch_filter_param=[300,50,100,0.5];
Psfs=Psfs_wo_yellow;
Error_counter=false(1,N);
parfor profile=1:N
    if max(yprofileYellow_norm(3:end-2,profile))>ch_filter_param(1)
        %         barcode_numbers_list(profile)
        warning('off','signal:findpeaks:largeMinPeakHeight')
        [pks,Ylocs,w,p]=findpeaks(yprofileYellow_norm(:,profile),"NPeaks",3,"SortStr","descend","MinPeakHeight",ch_filter_param(1),"MinPeakProminence",ch_filter_param(2),"MinPeakDistance",2.5,"MinPeakWidth",1.5,"WidthReference","halfheight");
%         findpeaks(yprofileYellow_norm(:,profile),"NPeaks",3,"SortStr","descend","MinPeakHeight",ch_filter_param(1),"MinPeakProminence",ch_filter_param(2),"MinPeakDistance",2.5,"MinPeakWidth",1.5,"WidthReference","halfheight","Annotate","extents");
        warning('on','signal:findpeaks:largeMinPeakHeight')
%         [~,Xlocs]=findpeaks(xprofileYellow(:,profile),"NPeaks",1,"SortStr","descend","MinPeakHeight",ch_filter_param(1),"MinPeakProminence",ch_filter_param(2),"MinPeakDistance",3);
        Xlocs=mean(Xrange(profile,:));
    else
        Ylocs=[];
        Xlocs=[];
    end
    if ~isempty(Ylocs)&&~isempty(Xlocs)
        n_rec_peaks=length(w);
        if sum(w>MinWidthFor2PeaksAnalysis)&&n_rec_peaks<3
            %             barcode_numbers_list(profile)
            if sum(w>MinWidthFor2PeaksAnalysis)==2
                    w(w~=max(w))=MinWidthFor2PeaksAnalysis-.1;
            end
            try
                [fit_pks,fit_locs,fit_w,r2]=doublePeakExtraction(yprofileYellow_norm(:,profile),n_rec_peaks,Ylocs,pks,w,MinWidthFor2PeaksAnalysis,barcode_numbers_list(profile),2);
            catch ME
                warning('off','backtrace')
                warning(ME.identifier,'%s problem with double gauss fitting in barcode %d channel %d index %d',ME.message,barcode_numbers_list(profile), 2,profile);
                warning('on','backtrace')
                Error_counter(profile)=true;
                r2=0;
            end
            if r2>0.85
                Ylocs=fit_locs;
            end

        end

        psfYellow=[psfFit_Image(image_stack_Yellow(:,:,profile),[repmat(Xlocs,1,length(Ylocs));Ylocs'],[],true,[],2);repmat(2,1,length(Ylocs))];
        if sum( psfYellow(5,:)>4|psfYellow(5,:)<0.5)
            psfYellow(:, psfYellow(5,:)>3|psfYellow(5,:)<0.5)=[];
        end
        if sum( psfYellow(4,:)==0)
            ind0=find(psfYellow(4,:)==0);
            for i=1:length(ind0)
                xpix=round(psfYellow(1,ind0(i)));
                ypix=round(psfYellow(2,ind0(i)));
                lb=max(xpix-3,1);
                rb=min(xpix+3,im_size(2));
                ub=min(ypix+3,im_size(1));
                db=max(ypix-3,1);
                psfYellow(4,ind0(i))= mean([image_stack_Yellow(ub,lb:rb,profile),image_stack_Yellow(db,lb:rb,profile),...
                    image_stack_Yellow(db:ub,lb,profile)',image_stack_Yellow(db:ub,rb,profile)']);
            end
        end
        if sum( psfYellow(3,:)<ch_filter_param(3))
            psfYellow(:, psfYellow(3,:)<ch_filter_param(3))=[];
        end
        if sum(psfYellow(3,:)./psfYellow(4,:)<ch_filter_param(4))
            psfYellow(:,psfYellow(3,:)./psfYellow(4,:)<ch_filter_param(4))=[];
        end
        if sum(psfYellow(1,:)<2.5|psfYellow(1,:)>(im_size(2)-1.5))
            psfYellow(:,psfYellow(1,:)<2.5|psfYellow(1,:)>(im_size(2)-1.5))=[];
        end
        if sum(psfYellow(2,:)<2.5|psfYellow(2,:)>(im_size(1)-1.5))
            psfYellow(:,psfYellow(2,:)<2.5|psfYellow(2,:)>(im_size(1)-1.5))=[];
        end
        if length(psfYellow(2,:))>1
            if sum(diff(sort(psfYellow(2,:),'ascend'))<=minYellowPeakDist)
                psfYellow=sortrows(psfYellow',2)';
                while(sum(diff(sort(psfYellow(2,:),'ascend'))<=minYellowPeakDist))
                    ind_minDist=find(diff(psfYellow(2,:))==min(diff(psfYellow(2,:))));
                    psfYellow(:,ind_minDist)=mean(psfYellow(:,ind_minDist:ind_minDist+1),2);
                    psfYellow(:,ind_minDist+1)=[];
                end
            end
        end
        Psfs{profile}=[Psfs{profile} psfYellow];
    end

    Psfs{profile}= sortrows(Psfs{profile}',2)';


    send(D,[]);
end

delete(wb);
Error_counter=sum(Error_counter,"all");

end

