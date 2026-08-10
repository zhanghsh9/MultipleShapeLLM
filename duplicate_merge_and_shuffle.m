%% Merge + shuffle with File Lists + Balancing Logic
clear all;

%% 1. Configuration: Add your file paths here
% You can add as many lines as you want inside the brackets [ ... ]

train_files = [ 
    "C:\Users\zhanghsh\OneDrive - The Pennsylvania State University\CEARL\proj4_random_shape_chiral_metasurface\other_datasets\An_data\cylinder_real_imag_train.json";
    "C:\Users\zhanghsh\OneDrive - The Pennsylvania State University\CEARL\proj4_random_shape_chiral_metasurface\other_datasets\An_data\H_real_imag_train.json";
    "C:\Users\zhanghsh\OneDrive - The Pennsylvania State University\CEARL\proj4_random_shape_chiral_metasurface\other_datasets\Adibnia_data\AOPS_T1_T2_train.json";
    "C:\Users\zhanghsh\OneDrive - The Pennsylvania State University\CEARL\proj4_random_shape_chiral_metasurface\other_datasets\Sullivan_data\micropyramids_train.json";
    "C:\Users\zhanghsh\OneDrive - The Pennsylvania State University\CEARL\proj4_random_shape_chiral_metasurface\other_datasets\Lu_data\ChatGPT_Metamaterial_train.json";
    "C:\Users\zhanghsh\OneDrive - The Pennsylvania State University\CEARL\proj4_random_shape_chiral_metasurface\other_datasets\Ag_nanorod\3rods_train.json";
    "C:\Users\zhanghsh\OneDrive - The Pennsylvania State University\CEARL\proj4_random_shape_chiral_metasurface\other_datasets\Ag_nanorod\4rods_train.json";
    "C:\Users\zhanghsh\OneDrive - The Pennsylvania State University\CEARL\proj4_random_shape_chiral_metasurface\other_datasets\Ag_nanorod\2rods_train.json";
    % "C:\Path\To\New_Dataset_train.json";  <-- Add future files here
];

test_files = [
    "C:\Users\zhanghsh\OneDrive - The Pennsylvania State University\CEARL\proj4_random_shape_chiral_metasurface\other_datasets\An_data\cylinder_real_imag_test.json";
    "C:\Users\zhanghsh\OneDrive - The Pennsylvania State University\CEARL\proj4_random_shape_chiral_metasurface\other_datasets\An_data\H_real_imag_test.json";
    "C:\Users\zhanghsh\OneDrive - The Pennsylvania State University\CEARL\proj4_random_shape_chiral_metasurface\other_datasets\Adibnia_data\AOPS_T1_T2_test.json";
    "C:\Users\zhanghsh\OneDrive - The Pennsylvania State University\CEARL\proj4_random_shape_chiral_metasurface\other_datasets\Sullivan_data\micropyramids_test.json";
    "C:\Users\zhanghsh\OneDrive - The Pennsylvania State University\CEARL\proj4_random_shape_chiral_metasurface\other_datasets\Lu_data\ChatGPT_Metamaterial_test.json";
    "C:\Users\zhanghsh\OneDrive - The Pennsylvania State University\CEARL\proj4_random_shape_chiral_metasurface\other_datasets\Ag_nanorod\3rods_test.json";
    "C:\Users\zhanghsh\OneDrive - The Pennsylvania State University\CEARL\proj4_random_shape_chiral_metasurface\other_datasets\Ag_nanorod\4rods_test.json";
    "C:\Users\zhanghsh\OneDrive - The Pennsylvania State University\CEARL\proj4_random_shape_chiral_metasurface\other_datasets\Ag_nanorod\2rods_test.json";
    % "C:\Path\To\New_Dataset_test.json";   <-- Add future files here
];

out_train_path  = ['merged_shuffled_train_', num2str(length(train_files)), '_geometries_unbalanced.json'];
out_test_path   = ['merged_shuffled_test_', num2str(length(train_files)), '_geometries_unbalanced.json'];

rng(1,"twister"); % Reproducible shuffle

%% 2. Read and Merge all files
fprintf("Reading %d training files...\n", length(train_files));
train_all = read_all_files(train_files);

fprintf("Reading %d testing files...\n", length(test_files));
test_all  = read_all_files(test_files);

%% 3. Balancing Logic (Oversampling) - Applied only to Training Set
fprintf("Analyzing training set geometry distribution...\n");

% Extract geometry types from all training objects
% (Using Regex to avoid parsing full JSON, preserving speed and text integrity)
train_geometries = strings(numel(train_all), 1);
for i = 1:numel(train_all)
    str = train_all{i};
    % Regex looks for: "geometry" : "VALUE"
    tokens = regexp(str, '"geometry"\s*:\s*"([^"]+)"', 'tokens');
    if ~isempty(tokens)
        train_geometries(i) = string(tokens{1}{1});
    else
        train_geometries(i) = "UNKNOWN";
        warning("Object %d has no geometry field.", i);
    end
end

% Count occurrences
[unique_geoms, ~, idx_map] = unique(train_geometries);
counts = accumarray(idx_map, 1);
[max_count, max_idx] = max(counts);

fprintf("Most counted geometry: '%s' with %d objects.\n", unique_geoms(max_idx), max_count);

% Duplicate objects to balance
train_balanced = {};

for i = 1:numel(unique_geoms)
    g_name = unique_geoms(i);
    g_count = counts(i);
    
    % Find all objects belonging to this geometry
    g_indices = find(strcmp(train_geometries, g_name));
    g_objs = train_all(g_indices);
    
    % Calculate multiplier n
    if g_count < 1e4
        n = min(round(max_count / g_count), 1);
    else
        n = min(round(max_count / g_count), 1);
    end
    if n < 1, n = 1; end
    
    fprintf("  Geometry '%s': count %d. Duplicating %d times (Target ~%d).\n", ...
        g_name, g_count, n, n*g_count);
    
    % Replicate the subset n times
    g_objs_expanded = repmat(g_objs, n, 1);
    
    % Append to the balanced list
    train_balanced = [train_balanced; g_objs_expanded]; %#ok<AGROW>
end

train_all = train_balanced; 
fprintf("Balancing complete. New training size: %d\n", numel(train_all));

%% 4. Shuffle and Write
% Shuffle order (object-level)
train_all = train_all(randperm(numel(train_all)));
test_all  = test_all(randperm(numel(test_all)));

% Write merged arrays
writeJsonArrayFromBlocks(out_train_path, train_all);
writeJsonArrayFromBlocks(out_test_path,  test_all);

fprintf("Done.\nFinal Train objects: %d\nFinal Test objects:  %d\n", numel(train_all), numel(test_all));


%% ---------- Local functions ----------

function all_blocks = read_all_files(fileList)
    % Helper to iterate over a list of paths and accumulate blocks
    all_blocks = {};
    for i = 1:length(fileList)
        fpath = fileList(i);
        fprintf("  -> Parsing: %s\n", fpath);
        blocks = readTopLevelObjectBlocks(fpath);
        all_blocks = [all_blocks; blocks]; %#ok<AGROW>
    end
end

function blocks = readTopLevelObjectBlocks(pathStr)
    txt = fileread(pathStr);

    % Find the top-level array boundaries
    i0 = find(txt=='[', 1, 'first');
    i1 = find(txt==']', 1, 'last');
    if isempty(i0) || isempty(i1) || i1 <= i0
        error("File does not look like a top-level JSON array: %s", pathStr);
    end

    blocks = {};
    inStr = false;
    esc = false;
    depth = 0;
    objStart = -1;

    for k = i0:i1
        ch = txt(k);

        if inStr
            if esc
                esc = false;
            else
                if ch == '\'
                    esc = true;
                elseif ch == '"'
                    inStr = false;
                end
            end
        else
            if ch == '"'
                inStr = true;

            elseif ch == '{'
                if depth == 0
                    objStart = k;
                end
                depth = depth + 1;

            elseif ch == '}'
                depth = depth - 1;
                if depth == 0 && objStart > 0
                    objEnd = k;
                    % Store EXACT object text
                    blocks{end+1,1} = txt(objStart:objEnd); %#ok<AGROW>
                    objStart = -1;
                end
            end
        end
    end

    if isempty(blocks)
        error("No top-level objects found in: %s", pathStr);
    end
end

function writeJsonArrayFromBlocks(pathStr, blocks)
    fid = fopen(pathStr,'w');
    if fid < 0, error("Cannot open output file: %s", pathStr); end

    fwrite(fid, "[", "char");
    fwrite(fid, newline, "char");

    for i = 1:numel(blocks)
        fwrite(fid, blocks{i}, "char"); 
        if i < numel(blocks)
            fwrite(fid, ",", "char");
        end
        fwrite(fid, newline, "char");
    end

    fwrite(fid, "]", "char");
    fwrite(fid, newline, "char");
    fclose(fid);
end