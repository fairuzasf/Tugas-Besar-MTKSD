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

% Mengambil data waktu dan polutan di satu titik (titik ke-1 misal)
x_data = data{:, 1};       % Hari ke-
y_data = data{:, 2};       % PM2.5 di titik 1

% Mencari titik puncak polusi (maksimum)
[max_val, max_idx] = max(y_data);
x0 = x_data(max_idx);      % Hari saat puncak terjadi

fprintf('Titik puncak polusi terjadi pada hari ke-%d dengan PM2.5 = %.2f\n', x0, max_val);

% Membuat window (jendela data) di sekitar titik puncak
% Mengambil contoh 10 hari sebelum dan 10 hari sesudah puncak
win_size = 10;
idx_start = max(1, max_idx - win_size);
idx_end = min(length(x_data), max_idx + win_size);

x_win = x_data(idx_start:idx_end);
y_win = y_data(idx_start:idx_end);

% Membuat Fungsi Kontinu f(x) menggunakan Polinomial
% Kita fit dengan polinomial orde 8 agar kurvanya mulus dan bisa diturunkan sampai orde 7
p = polyfit(x_win, y_win, 8); 

% Menghitung Turunan Analitik dari Polinomial p(x)
p1 = polyder(p);   % Turunan ke-1
p2 = polyder(p1);  % Turunan ke-2
p3 = polyder(p2);  % Turunan ke-3
p4 = polyder(p3);  % Turunan ke-4
p5 = polyder(p4);  % Turunan ke-5
p6 = polyder(p5);  % Turunan ke-6
p7 = polyder(p6);  % Turunan ke-7

% Mengevaluasi nilai turunan tepat di titik puncak (x0)
d0 = polyval(p, x0);
d1 = polyval(p1, x0);
d2 = polyval(p2, x0);
d3 = polyval(p3, x0);
d4 = polyval(p4, x0);
d5 = polyval(p5, x0);
d6 = polyval(p6, x0);
d7 = polyval(p7, x0);

% Membangun Deret Taylor Orde 3, 5, dan 7
dx = x_win - x0; % (x - x0)

% Taylor Orde 3
T3 = d0 + d1.*dx + (d2/factorial(2)).*dx.^2 + (d3/factorial(3)).*dx.^3;

% Taylor Orde 5 (Melanjutkan dari Taylor 3)
T5 = T3 + (d4/factorial(4)).*dx.^4 + (d5/factorial(5)).*dx.^5;

% Taylor Orde 7 (Melanjutkan dari Taylor 5)
T7 = T5 + (d6/factorial(6)).*dx.^6 + (d7/factorial(7)).*dx.^7;

% 8. Menghitung Galat Aproksimasi Analitik (Absolute Error)
% Galat dihitung terhadap fungsi kontinu f(x) yang kita buat
f_true = polyval(p, x_win);
err3 = abs(f_true - T3);
err5 = abs(f_true - T5);
err7 = abs(f_true - T7);

% Visualisasi nihh

% Grafik pertamaa: Perbandingan Fungsi dan Deret Taylor
figure;
plot(x_win, y_win, 'ko', 'DisplayName', 'Data Observasi Asli'); hold on;
plot(x_win, f_true, 'k-', 'LineWidth', 2, 'DisplayName', 'Fungsi Hampiran f(x)');
plot(x_win, T3, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Taylor Orde 3');
plot(x_win, T5, 'g-.', 'LineWidth', 1.5, 'DisplayName', 'Taylor Orde 5');
plot(x_win, T7, 'b:', 'LineWidth', 1.5, 'DisplayName', 'Taylor Orde 7');

% Menandai titik puncaknyaa
plot(x0, max_val, 'm*', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Titik Puncak (x_0)');

legend('Location', 'best');
title('Aproksimasi Deret Taylor di Sekitar Titik Puncak PM2.5');
xlabel('Hari Ke-'); 
ylabel('Konsentrasi PM2.5');
grid on;

% Grafik keduaaa: Perbandingan Galat
figure;
plot(x_win, err3, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Galat Orde 3'); hold on;
plot(x_win, err5, 'g-', 'LineWidth', 1.5, 'DisplayName', 'Galat Orde 5');
plot(x_win, err7, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Galat Orde 7');

legend('Location', 'best');
title('Perbandingan Galat Analitik Deret Taylor');
xlabel('Hari Ke-'); 
ylabel('Galat Absolut (|f(x) - T_n(x)|)');
grid on;

%% Bagian 5

%% Bagian 6

%% Bagian 7
