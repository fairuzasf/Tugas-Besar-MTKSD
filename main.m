clear; clc;

% Membaca dataset dari file CSV
data = readtable('DataSet_KualitasUdara_K1.csv', ...
    'VariableNamingRule', 'preserve');
% Mengekstrak  data (kolom 2 sampai 97)
A = table2array(data(:, 2:end));

%% BAGIAN 1
% Menampilkan ukuran matriks diawal untuk memverifikasi 
fprintf('BAGIAN 1: Matriks A dan Normalisasi Z-Score\n');
fprintf('Ukuran matriks A: %d baris x %d kolom\n', size(A, 1), size(A, 2));
[m, n] = size(A); % m = 365 (baris), n = 96 (kolom)

mean_j = (ones(1,m) * A) / m; % menghitung mean dari setiap kolom menggunakan operasi
                              % matriks dengan menjumlahkan semua baris tiap kolom, lalu dibagi m
simpangan = A - ones(m,1) * mean_j; % Menghtung simpangan agar rata-rata tiap kolom = 0
varians = (ones(1,m) * (simpangan .^ 2)) / m; % Menghitung varians setiap kolom
stddv = sqrt(varians); % Menghitung standar deviasi
% Mulai menghitung z_Score
Z = simpangan ./ (ones(m,1) * stddv);
fprintf('Ukuran matriks Z (setelah normalisasi): %d baris x %d kolom\n', size(Z,1), size(Z,2));
% verifikasi hasil normalisasi
mean_Z    = (ones(1,m) * Z) / m;
varians_Z   = (ones(1,m) * ((Z - ones(m,1)*mean_Z) .^ 2)) / m;
stddv_Z = sqrt(varians_Z);

fprintf('Verifikasi Normalisasi Z-Score:\n');
fprintf('Max nilai absolut mean Z : %.2e\n', max(abs(mean_Z)));
fprintf('Max selisih std Z dari 1 : %.2e\n', max(abs(stddv_Z - 1)));
fprintf('Min nilai Z : %.4f\n', min(Z(:)));
fprintf('Max nilai Z : %.4f\n', max(Z(:)));

%% Bagian 2

%% Bagian 3

%% Bagian 4

%% Bagian 5

%% Bagian 6

%% Bagian 7
