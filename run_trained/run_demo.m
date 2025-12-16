% Run Trained Model Demo in MATLAB
% This script demonstrates how to invoke the Python inference script from MATLAB.
% Since the model is a PyTorch (.pth) file, running it natively in MATLAB requires 
% conversion (e.g., to ONNX). The simplest integration is to call the Python script.

clc; close all; clear;

disp('----------------------------------------------------');
disp('Starting Inference Demo (calling Python backend)');
disp('----------------------------------------------------');

% Check if python is available
[status, ~] = system('python --version');
if status ~= 0
    error('Python is not found on the system path. Please install Python or adjust the path.');
end

% Command to run the python script
% Ensure we are in the correct directory or use absolute paths?
% Current directory is assumed to be the one containing this script.
command = 'python run_demo.py';

disp(['Executing command: ' command]);
[status, cmdout] = system(command);

% Check status
if status == 0
    disp('Success! Output from Python:');
    disp('----------------------------------------------------');
    disp(cmdout);
    disp('----------------------------------------------------');
    
    % The python script saves 'prediction_result.png'
    outputFile = 'prediction_result.png';
    
    if exist(outputFile, 'file')==2
        disp(['Displaying result: ' outputFile]);
        img = imread(outputFile);
        figure;
        imshow(img);
        title('Inference Result');
    else
        warning('Output file prediction_result.png was not found.');
    end
else
    error(['Python execution failed. Output:' 10 cmdout]);
end
