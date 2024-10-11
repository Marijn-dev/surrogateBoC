function fun_overview_h5file(file_name)
    % Check if the file exists
    if ~isfile(file_name)
        error('File does not exist.');
    end

    % Display file name
    fprintf('Overview of HDF5 file: %s\n', file_name);

    % Open the HDF5 file
    info = h5info(file_name);
    display_info(info, '/');

    function display_info(group, parent_name)
        % Display information about the current group
        fprintf('Group: %s\n', parent_name);

        % Display datasets in the current group
        for i = 1:length(group.Datasets)
            dataset = group.Datasets(i);
            dataset_name = fullfile(parent_name, dataset.Name);
            fprintf('  Dataset: %s\n', dataset_name);

            % Display dataset dimensions
            fprintf('    Dimensions: ');
            fprintf('%dx', dataset.Dataspace.Size);
            fprintf('\b\n');

            % Display dataset datatype
            fprintf('    Datatype: %s\n', dataset.Datatype.Class);

            % Display dataset attributes
            for j = 1:length(dataset.Attributes)
                attr = dataset.Attributes(j);
                fprintf('    Attribute: %s = ', attr.Name);
                disp(attr.Value);
            end
        end

        % Display groups in the current group
        for i = 1:length(group.Groups)
            subgroup = group.Groups(i);
            subgroup_name = fullfile(parent_name, subgroup.Name);
            display_info(subgroup, subgroup_name);
        end

        % Display attributes of the current group
        for i = 1:length(group.Attributes)
            attr = group.Attributes(i);
            fprintf('  Attribute: %s = ', attr.Name);
            disp(attr.Value);
        end
    end
end
