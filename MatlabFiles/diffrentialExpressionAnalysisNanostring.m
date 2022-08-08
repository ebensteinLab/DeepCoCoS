path=fullfile(pwd,"Data","nCounter");
load(fullfile(path,'NormalizedData.mat'));
varidx(1,:)=[1,5,8:11];%nCounter

varidx(2,:)=[1,5,12,14,16,18];%gt
varidx(3,:)=[1,5,13,15,17,19];%pred

NamesOrder={'HC1 nCounter','UC1 nCounter','UC2 nCounter','HC2 nCounter','HC1 GT','HC1 Pred','UC1 GT','UC1 Pred','UC2 GT','UC2 Pred','HC2 GT','HC2 Pred'};

Titles={'nCounter','Ground Truth','Prediction'};
cellMaxGenes=cell(1,3);
celldiffTable=cell(1,3);
for i=1:3
samplesToDisplay=varidx(i,3:end);
NamesToDisplay=NamesOrder(samplesToDisplay-7);
NormalizedDataForDiff=NormalizedData(NormalizedData.ClassName=="Endogenous",varidx(i,:));
diffTable = rnaseqde(NormalizedDataForDiff,[3,6],...
                     [4,5],VarianceLink="local",IDColumns="ProbeName");

celldiffTable{i}=diffTable;
sig = diffTable.AdjustedPValue < 0.05;
diffTableLocalSig = diffTable(sig,:);
diffTableLocalSig = sortrows(diffTableLocalSig,'AdjustedPValue');


[~,indMax]=sort(abs(diffTableLocalSig.Log2FoldChange),"descend");
absFoldMaxGenes=diffTableLocalSig(indMax,:);
top20GenesMax = absFoldMaxGenes(1:20,:);
cellMaxGenes{i}=top20GenesMax;

[foundMax,idxmax]=ismember(NormalizedDataForDiff.ProbeName,top20GenesMax.ProbeName);


end
gtMaxCount=sum(ismember(cellMaxGenes{2}(:,"ProbeName"),cellMaxGenes{1}(:,"ProbeName")));
predMaxCount=sum(ismember(cellMaxGenes{3}(:,"ProbeName"),cellMaxGenes{1}(:,"ProbeName")));
gtpredMaxCount=sum(ismember(cellMaxGenes{3}(:,"ProbeName"),cellMaxGenes{2}(:,"ProbeName")));
allMaxCount=size(intersect(cellMaxGenes{3}(:,"ProbeName"),intersect(cellMaxGenes{2}(:,"ProbeName"),cellMaxGenes{1}(:,"ProbeName"))),1);

%%
colors=[0 0.4470 0.7410
    0.8500 0.3250 0.0980
    0.9290 0.6940 0.1250];
figure(2)
clf
FCval=1.5;
Pval=0.05;
mostDiff20Table=cell(1,3);
for i=1:3
    sigdiffTable=celldiffTable{i}(celldiffTable{i}.AdjustedPValue<0.05,:);
    x=celldiffTable{i}.Log2FoldChange;
    y=celldiffTable{i}.AdjustedPValue;
    t=celldiffTable{i}.ProbeName;
    goodind=abs(x)>log2(FCval)&y<Pval;
    badind=abs(x)<log2(FCval)|y>Pval;
    displayLabelind=-log10(celldiffTable{i}.AdjustedPValue)>8;
    
    [~,FCind] = sort(abs(sigdiffTable.Log2FoldChange),'descend');
    displayLabelind=FCind(1:20);

h(i)=scatter(x(goodind),-log10(y(goodind)),30,colors(i,:),'filled','o','MarkerEdgeColor','flat','MarkerFaceColor','flat','MarkerFaceAlpha',.1,'MarkerEdgeAlpha',.5);
hold on
scatter(x(badind),-log10(y(badind)),30,[0.2 0.2 0.2],'filled','o','MarkerEdgeColor','flat','MarkerFaceColor','flat','MarkerFaceAlpha',.1,'MarkerEdgeAlpha',.5)
text(sigdiffTable.Log2FoldChange(displayLabelind)+.03,-log10(sigdiffTable.AdjustedPValue(displayLabelind)),sigdiffTable.ProbeName(displayLabelind),'FontSize',8,'Color',colors(i,:))
mostDiff20Table{i}=sigdiffTable(displayLabelind,:);
end

xline(log2(FCval),':',['FC=',num2str(FCval,2)],'LineWidth',2,'FontWeight','bold','FontSize',16);
xline(-log2(FCval),':',['FC=-',num2str(FCval,2)],'LineWidth',2,'FontWeight','bold','FontSize',16,'LabelHorizontalAlignment','left')
yline(-log10(Pval),':','Adj.P=0.05','LineWidth',2,'FontWeight','bold','FontSize',16)
legend([h(1) h(2) h(3)], {'nCounter','GT','Pred'})
% legend('boxoff')
xlabel('log2 Fold Change')
ylabel('-log10(Adjusted P-value)')
set(gca,'Fontsize',20,'FontWeight','bold')
hold off
%% Write to results to excel table
% sheetname={'nCounter','GT','Pred'};
% for i=1:3
% writetable(mostDiff20Table{i},fullfile(path,'Most20DiffGenesNormalizedNcounterRedults.xlsx','Sheet',sheetname{i});
% end