[data, dim, res] = readraw('Hip.vol');

[verts, norms] = recursive_marching_cubes(data, 90);

% Reshape T×9 into (3T)×3 vertex list
T = size(verts, 1);
verts_vlist = reshape(verts', 3, [])';  % Now (3T) × 3

% Close all figures
close all;  
figure;

% Faces are just sequential groups of 3
num_verts = size(verts_vlist, 1);  % = 3*T
faces = reshape(1:num_verts, 3, [])';

% Apply voxel resolution to get real-world coordinates (in mm)
verts_scaled = verts_vlist .* repmat(res, size(verts_vlist,1), 1);

% Then use verts_scaled in patch
p = patch('Faces', faces, 'Vertices', verts_scaled);

% Graphic settings
set(p, 'FaceColor', 'red', 'EdgeColor', 'none');
set(p, 'VertexNormals', norms); % We apply the calculated normal vectors
daspect([1 1 1]); % Respect the proportions
view(3); axis tight; camlight; lighting gouraud;
grid on;