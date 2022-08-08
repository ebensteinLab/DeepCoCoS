  A = [350 300 275]; I = [100 80 60 40];
    figure
    subplot(1,3,1), h1 = venn(A,I,'ErrMinMode','None');
    axis image,  title ('No 3-Circle Error Minimization')
    subplot(1,3,2), h2 = venn(A,I,'ErrMinMode','TotalError');
    axis image,  title ('Total Error Mode')
    subplot(1,3,3), h3 = venn(A,I,'ErrMinMode','ChowRodgers');
    axis image, title ('Chow-Rodgers Mode')
    set([h1 h2], 'FaceAlpha', 0.6)