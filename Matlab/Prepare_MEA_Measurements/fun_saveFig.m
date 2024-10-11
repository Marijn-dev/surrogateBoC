function [] = fun_saveFig(fig_name)
global pm

saveas(gcf, [pm.save.folder fig_name '.eps'], 'epsc');
% saveas(gcf, [pm.save.folder fig_name '.png']);

disp(['Figure saved as: ', fig_name]);

end