path=fullfile(pwd,"Data","Rosalind");
deepCoCoSGTonly=importdata(fullfile(path,'deepCoCoS_GT only.mat'));
deepCoCoSnCounteronly=importdata(fullfile(path,'deepCoCoS_nCounter only.mat'));
deepCoCoSPredOnly=importdata(fullfile(path,'deepCoCoS_Pred Only.mat'));
celldiffTable={deepCoCoSnCounteronly,deepCoCoSGTonly,deepCoCoSPredOnly};

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
    sigdiffTable=celldiffTable{i}(celldiffTable{i}.pvalue<Pval,:);
    sigdiffTable=celldiffTable{i};
    x=celldiffTable{i}.log2FoldChange;
    y=celldiffTable{i}.pvalue;
    t=celldiffTable{i}.Symbol;
    goodind=abs(x)>log2(FCval)&y<Pval;
    badind=abs(x)<log2(FCval)|y>Pval;
%     displayLabelind=-log10(celldiffTable{i}.pvalue)>8;
    
    [~,FCind] = sort(abs(sigdiffTable.log2FoldChange),'descend');
    displayLabelind=FCind(1:20);

h(i)=scatter(x(goodind),-log10(y(goodind)),30,colors(i,:),'filled','o','MarkerEdgeColor','flat','MarkerFaceColor','flat','MarkerFaceAlpha',.1,'MarkerEdgeAlpha',.5)
hold on
scatter(x(badind),-log10(y(badind)),30,[0.2 0.2 0.2],'filled','o','MarkerEdgeColor','flat','MarkerFaceColor','flat','MarkerFaceAlpha',.1,'MarkerEdgeAlpha',.5)
text(sigdiffTable.log2FoldChange(displayLabelind)+.05,-log10(sigdiffTable.pvalue(displayLabelind)),sigdiffTable.Symbol(displayLabelind),'FontSize',8,'Color',colors(i,:))
mostDiff20Table{i}=sigdiffTable(displayLabelind,:);
end

xline(log2(FCval),':',['FC=',num2str(FCval,2)],'LineWidth',2,'FontWeight','bold','FontSize',16);
xline(-log2(FCval),':',['FC=-',num2str(FCval,2)],'LineWidth',2,'FontWeight','bold','FontSize',16,'LabelHorizontalAlignment','left')
yline(-log10(Pval),':',['P.Val=',num2str(Pval,2)],'LineWidth',2,'FontWeight','bold','FontSize',16)
legend([h(1) h(2) h(3)], {'nCounter','GT','Pred'})
% legend('boxoff')
xlabel('log2 Fold Change')
ylabel('-log10(P-value)')
set(gca,'Fontsize',20,'FontWeight','bold')
hold off
%%

gtMaxCount=sum(ismember(mostDiff20Table{2}(:,"Symbol"),mostDiff20Table{1}(:,"Symbol")))
predMaxCount=sum(ismember(mostDiff20Table{3}(:,"Symbol"),mostDiff20Table{1}(:,"Symbol")))
gtpredMaxCount=sum(ismember(mostDiff20Table{3}(:,"Symbol"),mostDiff20Table{2}(:,"Symbol")))
allMaxCount=size(intersect(mostDiff20Table{3}(:,"Symbol"),intersect(mostDiff20Table{2}(:,"Symbol"),mostDiff20Table{1}(:,"Symbol"))),1)
%% Write to excel file
% sheetname={'nCounter','GT','Pred'};
% for i=1:3
% writetable(mostDiff20Table{i},fullfile(path,'Most20DiffGenesCorrected - Rosalind.xlsx'),'Sheet',sheetname{i});
% end