function fun_overview_print_h5file(file_name, output_file)
    % Check if the file exists
    if ~isfile(file_name)
        error('File does not exist.');
    end

    % Open the output file
    fid = fopen(output_file, 'w');
    if fid == -1
        error('Cannot open output file.');
    end

    % Display file name
    fprintf(fid, 'Overview of HDF5 file: %s\n', file_name);

    % Open the HDF5 file
    info = h5info(file_name);
    display_info(info, '/', fid);

    % Close the output file
    fclose(fid);

    function display_info(group, parent_name, fid)
        % Display information about the current group
        fprintf(fid, 'Group: %s\n', parent_name);

        % Display datasets in the current group
        for i = 1:length(group.Datasets)
            dataset = group.Datasets(i);
            dataset_name = fullfile(parent_name, dataset.Name);
            fprintf(fid, '  Dataset: %s\n', dataset_name);

            % Display dataset dimensions
            fprintf(fid, '    Dimensions: ');
            fprintf(fid, '%dx', dataset.Dataspace.Size);
            fprintf(fid, '\b\n');

            % Display dataset datatype
            fprintf(fid, '    Datatype: %s\n', dataset.Datatype.Class);

            % Display dataset attributes
            for j = 1:length(dataset.Attributes)
                attr = dataset.Attributes(j);
                fprintf(fid, '    Attribute: %s = ', attr.Name);
                disp_attr_value(fid, attr.Value);
            end
        end

        % Display groups in the current group
        for i = 1:length(group.Groups)
            subgroup = group.Groups(i);
            subgroup_name = fullfile(parent_name, subgroup.Name);
            display_info(subgroup, subgroup_name, fid);
        end

        % Display attributes of the current group
        for i = 1:length(group.Attributes)
            attr = group.Attributes(i);
            fprintf(fid, '  Attribute: %s = ', attr.Name);
            disp_attr_value(fid, attr.Value);
        end
    end

    function disp_attr_value(fid, value)
        if isnumeric(value) || islogical(value)
            fprintf(fid, '%g\n', value);
        elseif ischar(value)
            fprintf(fid, '%s\n', value);
        elseif iscell(value)
            fprintf(fid, '{');
            for k = 1:length(value)
                disp_attr_value(fid, value{k});
                if k < length(value)
                    fprintf(fid, ', ');
                end
            end
            fprintf(fid, '}\n');
        else
            fprintf(fid, 'Unsupported data type\n');
        end
    end
end
