%Latex interpreter
set(0, 'DefaultTextInterpreter', 'LaTeX');
set(0, 'DefaultLegendInterpreter', 'LaTeX');

% Set default text sizes for all subsequent plots
set(0, 'DefaultAxesFontSize', 16);
set(0, 'DefaultTextFontSize', 16);
set(0, 'DefaultLegendFontSize', 16);
% set(0, 'DefaultAxesTitleFontSize', 14);

% %Grid on
set(0, 'DefaultAxesXGrid', 'on');
set(0, 'DefaultAxesYGrid', 'on');

% %Marker layout
set(0, 'DefaultLineLineWidth', 1.5);
% set(0, 'DefaultLineMarkerSize', 8);

% Set default figure size and position
% defaultFigureSize = [800, 600];
% set(0, 'DefaultFigurePosition', [100, 100, defaultFigureSize]);
screenSize = get(0, 'ScreenSize');
