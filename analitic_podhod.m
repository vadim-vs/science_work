function mainfun

scrsz = get(0,'ScreenSize');
Form1 = figure('MenuBar', 'None',...
    'Name', 'Трассировка оптической системы',...
    'Position',[scrsz(3)/4 scrsz(4)/3 scrsz(3)/2 scrsz(4)/1.8],...
    'NumberTitle','Off')

[countLenses,lambda_inc,n_element,distance1,distance2,diff_order,diameter,order,lambda_base,focus,distance,scalecoeff, pow, flatSc, pointSc, isHarmonic, numberRays] = Show_Form();

%Добавление графика распространения лучей

ax_1 = axes; %привязываем ось к форме
set(ax_1,'Position',[0.1 0.085 0.85 0.4]); %установили на форме позицию и габариты
grid on;
xlabel('Расстояние, м');

global filename_new;
global intersection_parameters;
global g_vector_lens;
global g_vector_point;
global cccc 
cccc = 0;

global kk;
kk = 1;

global label1;
label1 = 0;

global state;
state = 0;

global regime;
regime = 0;

global filename_default;
filename_default = 'white1.bmp'; %по умолчанию будет подаваться белое изображение

global massive_n_elements;
global massive_diameters;
global massive_orders;
global massive_lambda_bases;
global massive_focuses;
global massive_distances;
global massive_powers;

%добавление названия и полей для ввода значений
function [countLenses,lambda_inc,n_element,distance1,distance2,diff_order,diameter,order,lambda_base,focus,distance,scalecoeff, pow, flatSc, pointSc, isHarmonic, numberRays] = Show_Form()
    
uicontrol('Style','Text',... %надпись "Кол-во линз"
     'String','Кол-во линз',... 
     'Position',[350,550,90,20]);
 
countLenses = uicontrol('Style','Edit',... %количество линз для пострения
    'String','1',...
    'Position',[450,550,40,20]);

uicontrol('Style','Text',... %надпись "Падающая длина волны (нм)"
     'String','Падающая длина волны (нм)',... 
     'Position',[350,510,90,30]);
 
lambda_inc = uicontrol('Style','Edit',... %длины волны, падающая на элемент в нм
    'String','532',...
    'Position',[450,515,40,20]);

uicontrol('Style','Text',... %надпись "Показатель преломления элемента"
     'String','Показатель преломления элемента',... 
     'Position',[350,450,90,50]);

n_element = uicontrol('Style','Edit',... %показатель преломления элемента
    'String','1.5',...
    'Position',[450,470,40,20]);

uicontrol('Style','Text',... %надпись "Расстояние вдоль оси (м)"
     'String','Расстояние вдоль оси (м)',... 
     'Position',[350,410,90,30]);

distance1 = uicontrol('Style','Edit',... %первая дистанция для ФРТ и для одной линзы плоскость рассмотрения в м
    'String','1.2',...
    'Position',[450,415,40,20]);

uicontrol('Style','Text',... %надпись "до"
     'String','до',... 
     'Position',[495,417,20,15]);
 
distance2 = uicontrol('Style','Edit',... %вторая дистанция для ФРТ в м
    'String','1.4',...
    'Position',[520,415,40,20]);

uicontrol('Style','Edit',... %надпись "Порядок дифракции"
     'String','Порядок дифракции',... 
     'Position',[350,382,120,18]);
 
diff_order = uicontrol('Style','Edit',... %надпись "Порядок дифракции"
     'String','0',... 
     'Position',[475,383.5,30,15]);
 
uicontrol('Style','Text',... %надпись "Порядок дифракции"
     'String','Порядок дифракции',... 
     'Position',[350,382,120,18]);

numberRays = uicontrol('Style','Edit',... %надпись "Порядок дифракции"
     'String','0',... 
     'Position',[475,342,35,20]);
 
uicontrol('Style','Edit',... %надпись "Порядок дифракции"
     'String','Кол-во лучей',... 
     'Position',[350,338,120,30]);

%%%%%%
uicontrol('Style','Text',... %надпись "Диаметр линзы (м)"
     'String','Диаметр линзы (м)',... 
     'Position',[100,550,105,20]);

diameter = uicontrol('Style','Edit',... %диаметр линзы в метрах
    'String','0.01',...
    'Position',[215,550,40,20]);

uicontrol('Style','Text',... %надпись "Порядок линзы"
     'String','Порядок линзы',... 
     'Position',[100,520,105,20]);

order = uicontrol('Style','Edit',... %порядок гармонической линзы
    'String','10',...
    'Position',[215,520,40,20]);

uicontrol('Style','Text',... %надпись "Базовая длина волны (м)"
     'String','Базовая длина волны (нм)',... 
     'Position',[100,480,105,30]);

lambda_base = uicontrol('Style','Edit',... %базовая длина волны
    'String','550',...
    'Position',[215,485,40,20]);

uicontrol('Style','Text',... %надпись "Фокус (м)"
     'String','Фокус (м)',... 
     'Position',[100,450,105,20]);

focus = uicontrol('Style','Edit',... %фокус линзы в метрах
    'String','1.0',...
    'Position',[215,450,40,20]);

uicontrol('Style','Text',... %надпись "Плоскость рассмотрения (м)"
     'String','Плоскость рассмотрения (м)',... 
     'Position',[100,410,105,30]);

distance = uicontrol('Style','Edit',... %дистанция от начальной плоскости для 1-ой линзы
    'String','1.0',...
    'Position',[215,415,40,20]);

uicontrol('Style','Text',... %надпись "Коэф. масштабирования" (если > 1 - увеличение, < 1 - уменьшение)
     'String','Коэф. масштаб.',... 
     'Position',[100,386,105,15]);

scalecoeff = uicontrol('Style','Edit',... %коэффициент масштабирования
    'String','1.0',...
    'Position',[215,385,40,16]);

uicontrol('Style','Text',... %надпись "Коэф. масштабирования" (если > 1 - увеличение, < 1 - уменьшение)
     'String','Степень линзы',... 
     'Position',[100,358,105,15]);

pow = uicontrol('Style','Edit',... %коэффициент масштабирования
    'String','2.0',...
    'Position',[215,358,40,16]);

flatSc = uicontrol('Style','checkbox',... 
     'String','Плоская волна',... 
     'Position',[600,530,100,35]);
 
pointSc = uicontrol('Style','checkbox',... 
     'String','Точечный источник',... 
     'Position',[600,500,100,35]);
 
isHarmonic = uicontrol('Style','checkbox',... 
     'String','Гармоническая линза',... 
     'Position',[600,450,100,35]);



%добавление кнопок управления

uicontrol('Style','PushButton',...
    'String','Загрузка изображения',...
    'Position',[800,550,110,20],...
    'CallBack',@Load_File);

uicontrol('Style','PushButton',...
    'String','ФРТ',...
    'Position',[830,515,40,20],...
    'CallBack',@PSF);

uicontrol('Style','PushButton',...
    'String','Трассировка',...
    'Position',[809,450,90,50],...
    'CallBack',@Traycing);

uicontrol('Style','PushButton',... 
     'String','Очистка',... 
     'Position',[819,412,65,20],...
     'CallBack',@Deleting);
 
uicontrol('Style','PushButton',... 
     'String','Распр. интенсивности',... 
     'Position',[800,370,100,35],...
     'CallBack',@DistOfIntensity);
 
%отталкиваться лучше относительно размера линзы
uicontrol('Style','Text',... 
     'String','Граница вывода изображения, OY',... 
     'Position',[600,385,100,35]);
 
borderImage = uicontrol('Style','Edit',... %коэффициент масштабирования
    'String','0.01',...
    'Position',[600,345,100,35]);
 

%
end

%%-----ФУНКЦИИ, ИСПОЛЬЗУЕМЫЕ ПРИ РАСЧЁТАХ-----------%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%нахождение направляющего вектора изображения
function [g_vector_image] = Guide_Vector_Image(phase, I, count_x, count_y, dx, dy)
    
    grad_fi_x = zeros(count_x,count_y);
    grad_fi_y = zeros(count_x,count_y);
    
    %потом подумаю, как можно поиграться ещё, чтоб быстрее было
    for i = 1 : count_x - 1
        if(i ~= count_x)
            for j = 1 : count_y - 1
                if(j ~= count_y)
                grad_fi_x(i,j) = (phase(i+1,j) - phase(i,j)) / dx; %вычисление градиентов дискретной функции
                grad_fi_y(i,j) = (phase(i,j+1) - phase(i,j)) / dy;
                end
            end
        end
        if (i == count_x)
            for j = 1 : count_x - 1
                grad_fi_x(count_x,j) = (phase(count_x,j) - phase(count_x - 1,j)) / dx; %вычисление градиентов дискретной функции
            end
        end
        if (j == count_y)
            for j = 1 : count_y - 1
                grad_fi_y(j,count_y) = (phase(j,count_y) - phase(j,count_y - 1)) / dy; %вычисление градиентов дискретной функции
            end
        end
    end
    
    %для i, j = N
    grad_fi_x(count_x,count_y) = (phase(count_x,count_y) - phase(count_x - 1,count_y)) / dx;
    grad_fi_y(count_x,count_y) = (phase(count_x,count_y) - phase(count_x,count_y - 1)) / dy;
    
    %норма для направляющего вектора
    norm = (grad_fi_x + grad_fi_y + 1.0).^(1/2.); %вычислили общую норму для нахождения направляющего вектора
    
    %направляющий вектор изображения
    g_vector_image = struct('cos_x',grad_fi_x ./ norm,'cos_y',grad_fi_y ./ norm,'cos_z',- 1 ./ norm,'I',I,'phase',phase);
    
end

%сюда посылаем уже изменённые значения
function [intersection_parameters] = Intersection_Parameters(x, y, z, count_x, count_y, A, B, C, D, I, phase, g_vector)
    %проверка на одномерность массива
    [m,n] = size(x);
    
    if((m == 1) || (n == 1))
        %делаем координаты двумерными (возможно, потом обратная замена потребуется)
        [x,y] = Increasing_Dimension(x, y, count_x, count_y);
    end
    
    %подумать над размерностью
    t = - (A .* x + B .* y  + C .* z + D) ./ (A .* g_vector.cos_x + B .* g_vector.cos_y + C .* g_vector.cos_z);
    
    %вгоняем эти параметры в новую структуру для вычисления параметров
    %точек пересечения
    intersection_parameters = struct('x',x + g_vector.cos_x .* t,'y',y + g_vector.cos_y .* t,'z',z + g_vector.cos_z .* t,'I',I,'phase',phase);
        
end

%создание из 1-го мерного 2-мерного вектора для корректной векторизации
%вычислений
function [x_2, y_2] = Increasing_Dimension(x, y, count_x, count_y)
    x_2 = zeros(count_x, count_y);
    y_2 = zeros(count_x, count_y);
    
    for j = 1 : count_y
        for i = 1 : count_x
            x_2(i,j) = x(i);
        end
    end
    
    for i = 1 : count_x
        for j = 1 : count_y
            y_2(i,j) = y(j);
        end
    end
    
end

%вычисление направляющего вектора
function [g_vector_lens] = Guide_Vector_Lens2(intersection_parameters, input_g_vector, n_lens, n_medium, k_inc, f, power, lambda_base)
      %dx, dy - шаг
      %number_lens - текущий номер линзы           
    norm_vector_lens = analitic_gradient(intersection_parameters, k_inc, f, power, lambda_base, n_lens);
              
    proverka_norm = norm_vector_lens.cos_x .* norm_vector_lens.cos_x + norm_vector_lens.cos_y .* norm_vector_lens.cos_y + norm_vector_lens.cos_z .* norm_vector_lens.cos_z;
    
    g_vector_lens = Find_G_Vector(norm_vector_lens, input_g_vector, n_lens, n_medium);
    
    aa = g_vector_lens.cos_x .* g_vector_lens.cos_x + g_vector_lens.cos_y .* g_vector_lens.cos_y + g_vector_lens.cos_z .* g_vector_lens.cos_z;
          
end

%аналитический расчёт с произвольной степенью
function [normalCos] = analitic_gradient(intersection_parameters, k_inc, f, power, lambda_base, n_lens)
    %градиент по каждому направлению будет определён в точке i,j, 
    %которая будет иметь свои координаты x(i,j) и y(i,j)
    [size_x, size_y] = size(intersection_parameters.x);
    
    if(cccc == 0)
        sq = 1;
    else
        sq = -1;
    end
    
    sq = 1;
        
    grad_x = (sq * k_inc / f / 2) * power .* ((intersection_parameters.x).^2 + (intersection_parameters.y).^2).^(power / 2 - 1) .* intersection_parameters.x;
    grad_y = (sq * k_inc / f / 2) * power .* ((intersection_parameters.x).^2 + (intersection_parameters.y).^2).^(power / 2 - 1) .* intersection_parameters.y;
    
    lambda_base = str2num(get(lambda_base, 'String')) * 10^(-9);
    grad_x = (lambda_base / (2 * pi *(n_lens - 1))) .* grad_x;
    grad_y = (lambda_base / (2 * pi *(n_lens - 1))) .* grad_y;
    
    norm = (grad_x .* grad_x + grad_y .* grad_y + 1.0).^(1/2.0);
    
    
    %нашли направляющий вектор линзы  
    normalCos = struct('cos_x', grad_x ./ norm , 'cos_y', grad_y ./ norm , 'cos_z', -1 ./ norm);
    
    cccc = cccc + 1;
    
end

%функция для нахождения координат направляющего вектора
function [g_vector_lens] = Find_G_Vector(norm_vector_lens, input_g_vector, n_lens, n_medium) %n_medium - n1, n_lens - n2

      if(cccc ~= 0)
          %input_g_vector.cos_z = - input_g_vector.cos_z;
          %input_g_vector.cos_y = - input_g_vector.cos_y;
          %input_g_vector.cos_x = - input_g_vector.cos_x;
      end
      
      cos_psi1 =  - norm_vector_lens.cos_x .* input_g_vector.cos_x - norm_vector_lens.cos_y .* input_g_vector.cos_y - norm_vector_lens.cos_z .* input_g_vector.cos_z;
      
      sin_psi1 = (1 - (cos_psi1).^2).^(1/2.); 
      
      sin_psi2 = (n_medium ./ n_lens).* sin_psi1; %закон Снеллиуса на поверхности
      
      cos_psi2 = (1 - (sin_psi2).^2).^(1/2.); 

      cos_phi = cos_psi1 .* cos_psi2 + sin_psi1 .* sin_psi2;
      
      aq1 = cos_psi1 .* cos_psi1 + sin_psi1 .* sin_psi1;
      aq2 = cos_psi2 .* cos_psi2 + sin_psi2 .* sin_psi2;
      
      a = - cos_psi2 ./ norm_vector_lens.cos_z;      
      d = - norm_vector_lens.cos_x ./ norm_vector_lens.cos_z;
      f = - norm_vector_lens.cos_y ./ norm_vector_lens.cos_z;      
      e = (cos_phi - a .* input_g_vector.cos_z) ./ (input_g_vector.cos_y + input_g_vector.cos_z .* f);
      g = (input_g_vector.cos_x + input_g_vector.cos_z .* d) ./ (input_g_vector.cos_y + input_g_vector.cos_z .* f);
            
      %далее происходит решение квадратного уравнения
      a0 = 1 + g .* g + (d - f .* g).^2;
      b0 = - 2 * e .* g + 2 * (a + f .* e) .* (d - f .* g);
      c0 = (a + f .* e).^2 + e .* e - 1;
    
      D = b0 .* b0 -  4 * a0 .* c0;
     % if(abs(D) < 10^(-6))
      %    D = 0.0;
      %end
    
      root1 = (- b0 - D.^(1/2.0)) ./ (2 * a0);
      cz2 = (- b0 - D.^(1/2)) ./ (2 * a0); %возможно, нужно будет условие
      %поставить
    
      cos_x = real(root1); %x-coordinate
      cos_y = e - g .* cos_x; %y-coordinate
      cos_z = a + d .* cos_x + f .* cos_y; %z-coordinate
      
      aaa1 = cos_x(1,1) + cos_y(1,1) + cos_z(1,1);
      aaa = (cos_x .* cos_x + cos_y .* cos_y + cos_z .* cos_z) .^ (1/2.0);
      
      %нашли направляющий вектор линзы  
      g_vector_lens = struct('cos_x', cos_x , 'cos_y', cos_y , 'cos_z', cos_z);
      
      regime = regime + 1;
end 

function [g_vector_lens] = Find_G_Vector2(norm_vector_lens, input_g_vector, n_lens, n_medium) %n_medium - n1, n_lens - n2

      cos_psi1 =  - norm_vector_lens.cos_x .* input_g_vector.cos_x - norm_vector_lens.cos_y .* input_g_vector.cos_y - norm_vector_lens.cos_z .* input_g_vector.cos_z;
      
      sin_psi1 = (1 - (cos_psi1).^2).^(1/2.); 
      
      sin_psi2 = (n_medium ./ n_lens).* sin_psi1; %закон Снеллиуса на поверхности
      
      cos_psi2 = (1 - (sin_psi2).^2).^(1/2.); 

      cos_phi = cos_psi1 .* cos_psi2 + sin_psi1 .* sin_psi2;
      
      aq1 = cos_psi1 .* cos_psi1 + sin_psi1 .* sin_psi1;
      aq2 = cos_psi2 .* cos_psi2 + sin_psi2 .* sin_psi2;
      
      a = - cos_psi2 ./ norm_vector_lens.cos_z - (norm_vector_lens.cos_z .* cos_phi + input_g_vector.cos_z .* cos_psi2) ./ (input_g_vector.cos_z .* norm_vector_lens.cos_y - input_g_vector.cos_y .* norm_vector_lens.cos_z);    
      b = - norm_vector_lens.cos_x ./ norm_vector_lens.cos_z .* (input_g_vector.cos_y ./ norm_vector_lens.cos_z .* (norm_vector_lens.cos_y - norm_vector_lens.cos_x) + input_g_vector.cos_x - input_g_vector.cos_y) ./ (input_g_vector.cos_z .* norm_vector_lens.cos_y ./ norm_vector_lens.cos_z - input_g_vector.cos_y);
      c = (norm_vector_lens.cos_z .* input_g_vector.cos_x - input_g_vector.cos_z .* norm_vector_lens.cos_x) ./ (input_g_vector.cos_z .* norm_vector_lens.cos_y - input_g_vector.cos_y .* norm_vector_lens.cos_z);
      d = (cos_phi .* norm_vector_lens.cos_z + input_g_vector.cos_z .* cos_psi2) ./ (input_g_vector.cos_z .* norm_vector_lens.cos_y - input_g_vector.cos_y .* norm_vector_lens.cos_z);
                 
      %далее происходит решение квадратного уравнения
      a0 = 1 + b .* b + c .* c;
      b0 = - 2 * c .* d - 2 * a .* b;
      c0 = d .* d + a .* a - 1;
    
      D = b0 .* b0 -  4 * a0 .* c0;
     % if(abs(D) < 10^(-6))
      %    D = 0.0;
      %end
    
      root1 = (- b0 + D.^(1/2.0)) ./ (2 * a0);
      cz2 = (- b0 - D.^(1/2)) ./ (2 * a0); %возможно, нужно будет условие
      %поставить
    
      cos_x = real(root1); %x-coordinate
      cos_y = c .* cos_x - d; %y-coordinate
      cos_z = a - b .* cos_x; %z-coordinate
      
      aaa1 = cos_x(1,1) + cos_y(1,1) + cos_z(1,1);
      aaa = (cos_x .* cos_x + cos_y .* cos_y + cos_z .* cos_z) .^ (1/2.0);
      
      %нашли направляющий вектор линзы  
      g_vector_lens = struct('cos_x', cos_x , 'cos_y', cos_y , 'cos_z', cos_z);
      
      regime = regime + 1;
end 


%нахождение распределения интенсивности в координатах Х и Y (будем рассматривать 
%квадратные границы) и размерность зададим (целое число должно быть)
function [] = DistShow(intersection_parameters, N, image)
    
    %здесь нужно будет поработать над автоматическим масштабированием (сейчас просто константный масштаб)
    [count_x, count_y] = size(intersection_parameters.x);
    
    intensityMassive = zeros(count_x, count_y); %массив, в который будем записывать
       
        %получили координаты Х и Y с учётом масштаба
        x = N * intersection_parameters.x(:,1);
        y = N * (intersection_parameters.y(1,:))';
        
        y = real(y);
        x = real(x);

        %сначала выполним проверку на чётность и нечётность
        %размерности (для нормального нахождения середины)

        mid_x = round(count_x / 2.0);
        mid_y = round(count_y / 2.0);

        for i = 1 : count_x
            k = round(x(i));

            for j = 1 : count_y
                t = round(y(j));
                if(label1 == 1)
                    if(~isnan(t))
                        intensityMassive(mid_x + k, mid_y + t) = intensityMassive(mid_x + k, mid_y + t) + image(mid_x + k, mid_y + t);
                    end
                else
                    if(~isnan(t))
                        intensityMassive(mid_x + k, mid_y + t) = intensityMassive(mid_x + k, mid_y + t) + 1;
                    end
                end
            end 

        end
    
    %теперь нормализуем на уровень 255 для нормальной работы 
    %функции imshow
    maxElem = max(max(intensityMassive));
    minElem = min(min(intensityMassive));
    
    figure;
    imshow(intensityMassive, [0 1])      
    
end

%масштабирование изображения
function [] = Scaling(scalecoeff, image, left_border_x, right_border_x, left_border_y, right_border_y)
    
    %сделаем через plot, дальше надо будет подумать
    [countX, countY] = size(image.x);
    
    %запись матрицы в вектор
    x = reshape(image.x, 1, countX * countY);
    y = reshape(image.y, 1, countX * countY);
        
    %вот и само масштабирование (будем масштабировать от начального значения)
    x_range = [left_border_x/scalecoeff right_border_x/scalecoeff];
    y_range = [left_border_y/scalecoeff right_border_y/scalecoeff];
    
    figure;
    plot(x, y,'k.')
    axis([x_range y_range])
             
    
end

%обрезать изображения до круга
function [] = Get_Circle(current_image, radius_lens, x, y)
    [size_x, size_y] = size(current_image);
    
    for i = 1 : size_x
        for j = 1 : size_y
            if((x(i) * x(i) + y(j) * y(j))^(1/2.0) > radius_lens)
                current_image(i,j) = 0.0; %зачерним область
            end
        end
    end
    
    figure('Name', 'Выделение круглой области',...
        'NumberTitle','Off');
    
    imshow(current_image)
    
end


%%-----ФУНКЦИИ, АКТИВИРУЮЩИЕСЯ ПРИ НАЖАТИИ КНОПОК---%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%выбор определённого файла и сохранение пути для него для дальнейшей
%загрузки
function [] = Load_File(h,eventdata)
    [filename, pathfile] = uigetfile();
    label1 = 1;
    filename_new = strcat(pathfile,filename);
    if(filename == 0) %нажали cancel
        label1 = 0;
        filename_new = filename_default;
    end
end

%сюда будем возвращать параметры, вбитые по форме для каждой линзы
function [] = Init_Params(h,eventdata, n_element, diameter, order, lambda_base, focus, distance, scalecoeff, pow, flatSc, pointSc)
    
    countLenses = str2num(get(countLenses,'String'));
    lambda_inc = str2num(get(lambda_inc, 'String')) * 10^(-9);
    diff_order = str2num(get(diff_order, 'String'));
    
    massive_n_elements = zeros(1,countLenses);
    massive_diameters = zeros(1,countLenses);
    massive_orders = zeros(1,countLenses);
    massive_lambda_bases = zeros(1,countLenses);
    massive_focuses = zeros(1,countLenses);
    massive_distances = zeros(1,countLenses);
    massive_powers = zeros(1,countLenses);
    
    
    if(countLenses > 1)
        n_element_vector = zeros(1,countLenses);
        diameter_vector = zeros(1,countLenses);
        order_vector = zeros(1,countLenses);
        lambda_base_vector = zeros(1,countLenses);
        focus_vector = zeros(1,countLenses);
        distance_vector = zeros(1,countLenses);
        powers_vector = zeros(1,countLenses);
        
        i = 1; 
        k = 1; % для нажатия мыши
        
        while i ~= countLenses + 1 
            title_form = strcat('Параметры линзы № ',num2str(i));
            
            Form2 = figure('MenuBar', 'None',...
                'Name', title_form,...
                    'NumberTitle','Off',...
                    'Position',[400,250,300,200]);
   
            uicontrol('Style','Text',... %надпись "Падающая длина волны (нм)"
                 'String','Показатель преломления',... 
                 'Position',[10,165,80,30]);       

            n_element_vector(i) = uicontrol('Style','Edit',... %показатель преломления элемента
                'String','1.5',...
                'Position',[95,170,30,20]);
                
            uicontrol('Style','Text',... %надпись "Диаметр линзы (м)"
                 'String','Диаметр (м)',... 
                 'Position',[10,135,80,20]);

            diameter_vector(i) = uicontrol('Style','Edit',... %диаметр линзы в метрах
                'String','0.01',...
                'Position',[95,135,30,20]);

            uicontrol('Style','Text',... %надпись "Диаметр линзы (м)"
                 'String','Порядок линзы',... 
                 'Position',[10,105,80,20]);

            order_vector(i) = uicontrol('Style','Edit',... %диаметр линзы в метрах
                'String','10',...
                'Position',[95,105,30,20]);
            
            uicontrol('Style','Text',... %надпись "Диаметр линзы (м)"
                 'String','Порядок линзы',... 
                 'Position',[10,75,80,20]);
             
            powers_vector(i) = uicontrol('Style','Edit',... %диаметр линзы в метрах
                'String','2.0',...
                'Position',[95,75,30,20]);

            uicontrol('Style','Text',... %надпись "Диаметр линзы (м)"
                 'String','Длина волны баз (нм)',... 
                 'Position',[160,165,80,30]);

            lambda_base_vector(i) = uicontrol('Style','Edit',... %диаметр линзы в метрах
                'String','550',...
                'Position',[245,170,40,20]);

            uicontrol('Style','Text',... %надпись "Диаметр линзы (м)"
                 'String','Фокус (м)',... 
                 'Position',[160,135,80,20]);

            focus_vector(i) = uicontrol('Style','Edit',... %диаметр линзы в метрах
                'String','1.0',...
                'Position',[245,135,40,20]);

            uicontrol('Style','Text',... %надпись "Диаметр линзы (м)"
                 'String','Расстояние от пред. объекта (м)',... 
                 'Position',[160,83,80,42]);

            distance_vector(i) = uicontrol('Style','Edit',... %диаметр линзы в метрах
                'String','1.0',...
                'Position',[245,105,40,20]);
            
            if(i ~= countLenses)
                    uicontrol('Style','PushButton',...
                        'String','Далее',...
                        'Position',[110,15,80,40]);
                    
                    while mod(k,8) ~= 0
                        if(waitforbuttonpress() == 0)
                            k = k + 1;   
                        end
                    end
            else
                    uicontrol('Style','PushButton',...
                        'String','Выход',...
                        'Position',[110,15,80,40]);
                    
                    while mod(k,8) ~= 0
                        if(waitforbuttonpress() == 0)
                            k = k + 1;   
                        end
                    end
                    
            end
       
            massive_n_elements(i) = str2num(get(n_element_vector(i),'String'));
            massive_diameters(i) = str2num(get(diameter_vector(i),'String'));
            massive_orders(i) = str2num(get(order_vector(i),'String'));
            massive_lambda_bases(i) = str2num(get(lambda_base_vector(i),'String')) * 10^(-9);
            massive_focuses(i) = str2num(get(focus_vector(i),'String'));
            massive_distances(i) = str2num(get(distance_vector(i),'String'));
            massive_powers(i) = str2num(get(powers_vector(i),'String'));

            i = i + 1;
            k = k + 1;
        end    
        
    else    
    
        massive_n_elements = str2num(get(n_element,'String'));
        massive_diameters = str2num(get(diameter,'String'));
        massive_orders = str2num(get(order,'String'));
        massive_lambda_bases = str2num(get(lambda_base,'String')) * 10^(-9);
        massive_focuses = str2num(get(focus,'String'));
        massive_distances = str2num(get(distance,'String'));
        massive_powers = str2num(get(pow,'String'));
               
    end
    
    
end

%функция, которая будет производить расчёт всей системы и выкидывать
%картинку на выходе (и график в главном окне строить)
function [] = Traycing(h,eventdata)
    
    Init_Params(h,eventdata, n_element, diameter, order, lambda_base, focus, distance, scalecoeff, pow, flatSc, pointSc);
    
    %условимся загружать только .bmp изображения
    %если изображение не было загружено (на всякий случай)
    %если было загружено нормально, то так и так поменяется
    if (label1 == 0)
        filename_new = filename_default;
    end
    
    %подгружаем само изображение, расположенное по уже найденной директории
    image = imread(filename_new, 'bmp');
    
    %подгрузим чекеры в форме 1 - стоит галочка, 0 - не стоит (далее будут 
    %необходимы для выбора того, что падает на линзу)
    flatSc = get(flatSc,'Value');
    pointSc = get(pointSc, 'Value');
    isHarmonic = get(isHarmonic, 'Value'); %использование вместо простых линз гармонических
       
    scalecoeff = str2num(get(scalecoeff,'String'));
    diameter = str2num(get(diameter,'String'));
    pow = str2num(get(pow,'String'));
    
    count_x = size(image,1); %N - количество пикселей в направлении оси ОХ
    count_y = size(image,2); %M - количество пикселей в направлении оси ОY
            
    %волновое число падающей длины волны
    k_inc = 2 * pi / lambda_inc;    
    
    %зададим константу для изменения фазы сами (в дальнейшем, нужно будет это модифицировать)
    C = 2 * pi;
    
    %определим размер изображения (потом надо будет добавить на форму главную)
    size_image_x = diameter; % в метрах
    size_image_y = diameter; 
    
    x0 = linspace(-size_image_x, size_image_x, count_x);
    y0 = linspace(-size_image_y, size_image_y, count_y);
        
    dx0 = x0(2) - x0(1);
    dy0 = x0(2) - x0(1);
    
    %определим интенсивность и фазу
    I_image = double(image(:,:,1));
    
    phase = zeros(count_x, count_y); %%%%%!!!!
    phase = C + phase;
    
    %Get_Circle(I_image, radius_lens, x0, y0);
    
    if(flatSc == 1)
        %в этом случае плоский волновой фронт на входе
        g_vector_image = Guide_Vector_Image(phase, I_image, count_x, count_y, dx0, dy0);
    end
    
    len_distance = str2num(get(distance, 'String'));
    
    if(pointSc == 1)
         %в этом случае излучающая точка в качестве источника (необходимо задать стартовую точку для излучения)
         x00 = zeros(count_x, count_y);
         y00 = zeros(count_x, count_y);
         z00 = zeros(count_x, count_y);
         
         startPoint = struct('x', x00, 'y', y00, 'z', z00);
        
         g_vector_image = RadiationPoint(diameter, diameter, count_x, count_y, distance, startPoint);
    end
    
    %показатель преломления среды
    n_medium = 1.0;
    
    %необходимо задать параметры для корректного построения хода лучей на
    %графике главной формы
    y_vec = zeros(countLenses + 2, count_y); %+2 потому что ещё есть начальная плоскость и ещё плюс выходная плоскость
    z_vec = zeros(countLenses + 2, count_y);
    
    %задаём первоначальные координаты изображения
    if(pointSc)
        y_vec(1,:) = 0.0;
        z_vec(1,:) = 0.0;
    else
        y_vec(1,:) = y0(:);
        z_vec(1,:) = 0.0;
    end
    
    %для всех линз будем прогонять все параметры
    for i = 1 : countLenses
        
        x = linspace(-massive_diameters(i),massive_diameters(i),count_x);
        y = linspace(-massive_diameters(i),massive_diameters(i),count_y);
        
        dx = x(2) - x(1); %шаг
        dy = y(2) - y(1);
        
        %дифракционная эффективность i - ой линзы на k - ом порядке
        nu = sinc(massive_orders(i) * massive_lambda_bases(i) / lambda_inc - diff_order) * sinc(massive_orders(i) * massive_lambda_bases(i) / lambda_inc - diff_order);
        
        I = I_image;
        %интенсивность, с учётом диф. эффективности
        %I = nu * I_image;
        
        %если всего одна линза
        if(countLenses == 1)
            %len_distance = str2num(get(distance, 'String'));
            focal_plane = str2num(get(distance2, 'String'));
            order_lens = str2num(get(order, 'String'));
            
            A = 0.0;
            B = 0.0;
            C = 1.0;
            D = - len_distance;
            
            %находим координаты пересечения лучей с плоскостью гармонической
            %линзы, находящейся на расстоянии len_distance
            if(pointSc == 1)
                intersection_parameters = Intersection_Parameters(startPoint.x, startPoint.y, 0.0, count_x, count_y, A, B, C, D, I, phase, g_vector_image);
            else
                intersection_parameters = Intersection_Parameters(x0, y0, 0.0, count_x, count_y, A, B, C, D, I, phase, g_vector_image);
            end
            
            %будем выводить координаты пересечения с плоскостью линзы
            y_vec(2,:) = intersection_parameters.y(1,:);
            z_vec(2,:) = intersection_parameters.z(1,:);
                    
            %находим направляющий вектор лучей после линзы
            g_vector_lens = Guide_Vector_Lens2(intersection_parameters, g_vector_image, massive_n_elements, n_medium, k_inc, massive_focuses, pow, lambda_base);
            
            A = 0.0;
            B = 0.0;
            C = 1.0;
            D = - focal_plane;
            
            %находим координаты лучей на заданном расстоянии после линзы
            intersection_parameters = Intersection_Parameters(intersection_parameters.x, intersection_parameters.y, intersection_parameters.z, count_x, count_y, A, B, C, D, intersection_parameters.I, intersection_parameters.phase, g_vector_lens);
                
            %и выведем координаты на некотором расстоянии от линзы
            y_vec(3,:) = intersection_parameters.y(1,:);
            z_vec(3,:) = intersection_parameters.z(1,:);
            
            countPoints = 10000;
            %DistShow(intersection_parameters, countPoints);
            
        %если много линз и заполняем данные для первой линзы
        elseif((countLenses ~= 1)&&(i == 1))
            
            A = 0.0;
            B = 0.0;
            C = 1.0;
            D = - massive_distances(i);
            
            %находим координаты пересечения лучей с плоскостью гармонической
            %линзы, находящейся на расстоянии len_distance
            intersection_parameters = Intersection_Parameters(x0, y0, 0.0, count_x, count_y, A, B, C, D, I, phase, g_vector_image);
            
            %будем выводить координаты пересечения с плоскостью линзы
            y_vec(i + 1,:) = intersection_parameters.y(1,:);
            z_vec(i + 1,:) = intersection_parameters.z(1,:);          
                           
            %находим направляющий вектор лучей после линзы
            g_vector_lens = Guide_Vector_Lens2(intersection_parameters, g_vector_image, massive_n_elements(i), n_medium, k_inc, massive_focuses(i), pow, lambda_base);
                       
            A = 0.0;
            B = 0.0;
            C = 1.0;
            D = - massive_distances(i + 1);
            
            %находим координаты лучей на заданном расстоянии после линзы (или координаты лучей при пересечении с плоскостью)
            intersection_parameters = Intersection_Parameters(intersection_parameters.x, intersection_parameters.y, intersection_parameters.z, count_x, count_y, A, B, C, D, intersection_parameters.I, intersection_parameters.phase, g_vector_lens);
        
            %будем выводить координаты пересечения с плоскостью линзы
            y_vec(i + 2,:) = intersection_parameters.y(1,:);
            z_vec(i + 2,:) = intersection_parameters.z(1,:);
                   
        %если дошли до последней линзы до последнего элемента
        elseif(i == countLenses)
                                  
            %Guide_Vector_Lens2(intersection_parameters, input_g_vector, n_lens, n_medium, k_inc, f, power, lambda_base)
        
            %находим направляющий вектор лучей после линзы
            g_vector_lens = Guide_Vector_Lens2(intersection_parameters, g_vector_lens, massive_n_elements(i), n_medium, k_inc, massive_focuses(i), pow, lambda_base);
        
            A = 0.0;
            B = 0.0;
            C = 1.0;
            D = - str2num(get(distance2,'String'));
            
            %находим координаты лучей на заданном расстоянии после линзы (или координаты лучей при пересечении с плоскостью)
            intersection_parameters = Intersection_Parameters(intersection_parameters.x, intersection_parameters.y, intersection_parameters.z, count_x, count_y, A, B, C, D, intersection_parameters.I, intersection_parameters.phase, g_vector_lens);
            
            %будем выводить координаты пересечения с плоскостью линзы
            y_vec(countLenses + 2,:) = intersection_parameters.y(1,:);
            z_vec(countLenses + 2,:) = intersection_parameters.z(1,:);

            countPoints = 10000;
            %DistShow(intersection_parameters, countPoints, I);
            
            %aa=0;
        else
        
        %на промежуточных линзах
             
        %находим направляющий вектор лучей после линзы
        g_vector_lens = Guide_Vector_Lens2(intersection_parameters, g_vector_lens, massive_n_elements(i), n_medium, k_inc, massive_focuses(i), pow, lambda_base);
                 
        A = 0.0;
        B = 0.0;
        C = 1.0;
        D = - massive_distances(i + 1);
                    
        %находим координаты лучей на заданном расстоянии после линзы (или координаты лучей при пересечении с плоскостью)
        intersection_parameters = Intersection_Parameters(intersection_parameters.x, intersection_parameters.y, intersection_parameters.z, count_x, count_y, A, B, C, D, intersection_parameters.I, intersection_parameters.phase, g_vector_lens);
                 
        %будем выводить координаты пересечения с плоскостью линзы
        y_vec(i + 2,:) = intersection_parameters.y(1,:);
        z_vec(i + 2,:) = intersection_parameters.z(1,:);
                
        end
    
    end
    
    axes(ax_1);
    
    y1 = zeros(countLenses + 2,1);
    z1 = zeros(countLenses + 2,1);
    
    %теперь выберем количество лучей, которые будем выводить (если так и 
    %оставить 0, то будет выводить по дефолту)
    
    numberRays = ceil(str2num(get(numberRays,'String')));
    
    if(numberRays ~= 0)
        
       numberRays = numberRays + 1; %чтобы выводилось действительно то количество лучей, которые задали
       h = ceil(count_y / numberRays);
        
       for j = 1 : numberRays - 2
            for i = 1 : countLenses + 2
                y1(i) = y_vec(i,j * h);
                z1(i) = z_vec(i,j * h);   
            end
       
            plot(z1,y1);
            hold on;
       end
       
    else
        
       for j = 1:count_y
            for i = 1 : countLenses + 2
                y1(i) = y_vec(i,j);
                z1(i) = z_vec(i,j);   
            end
            
            plot(z1,y1);
            hold on;
       end
    end
   
    %Scaling(scalecoeff, intersection_parameters, -massive_diameters, massive_diameters, -massive_diameters, massive_diameters)
    
    %можно будет возвращать что-то отсюда для ФРТ (но над этим позже подумаем)
end

%нахождение распределения интенсивности вдоль оси распространения пучка
function [] = PSF(h,eventdata)
    %для тестирования возьмём 100 точек.Если всё норм, то изменим на 1000
    count_points = 2000;
    
    if (label1 == 0)
        filename_new = filename_default;
    end
    
    %подгружаем само изображение, расположенное по уже найденной директории
    image = imread(filename_new, 'bmp');
    
    count_x = size(image,1); %N - количество пикселей в направлении оси ОХ
    count_y = size(image,2); %M - количество пикселей в направлении оси ОY

    A = 0.0;
    B = 0.0;
    C = 1.0;
    
    %рассматриваемый участок
    z_look = linspace(-str2num(get(distance1,'String')),-str2num(get(distance2,'String')),count_points);
    len_z_look = length(z_look);
    
    %интенсивность в рассматриваемых точках z_look
    I_look = zeros(1,len_z_look);
    
    %если не был запущен до этого трейсинг
    
    if(state == 0)   
        
        Traycing(h,eventdata);
        
    end    
        
        for k = 1 : len_z_look
                          
            intersection_parameters = Intersection_Parameters(intersection_parameters.x, intersection_parameters.y, intersection_parameters.z, count_x, count_y, A, B, C, z_look(k), intersection_parameters.I, intersection_parameters.phase, g_vector_lens);
            
            case1 = (abs(intersection_parameters.y))<= 2* 10^(-5);
            case2 = (abs(intersection_parameters.x))<= 2* 10^(-5);
            
            case3 = case1 .* case2;
            
            q = 0;
            
            I = 0;
            for i = 1 : count_x
                for j = 1 : count_y
                    if(case3(i,j) == 1)
                        q = q + 1;
                        I = I + intersection_parameters.I(i,j);
                    end
                end
            end
            
            I_look(k) =  I_look(k) + I;
            
            disp(k)
        end
        
        I_max = max(I_look);
        I_look = I_look ./ I_max;
        
    
    %нахождение диаметра пятна по полуспаду интенсивности
       
    [I_maxx, t] = max(I_look);
    count = t;
            
    while (I_look(count) > (I_maxx/2))
        count = count + 1;
    end
            
    spot_diameter = 2 * abs(z_look(t) - z_look(count));
    
    disp('Диаметр пятна (м) равен')
    disp(spot_diameter)
    %
    
    %вывод графика ФРТ на экран
    figure('Name','ФРТ');
    plot(z_look, I_look);
    grid on;
    title('Интенсивность');
    xlabel('Расстояние вдоль оси, м');
    
end

%функция рассения точки вдоль (вдоль радиуса)
function [] = DistOfIntensity(h,eventdata)
    %для тестирования возьмём 100 точек.Если всё норм, то изменим на 1000
    count_points = 2000;
    
    if (label1 == 0)
        filename_new = filename_default;
    end
    
    %подгружаем само изображение, расположенное по уже найденной директории
    image = imread(filename_new, 'bmp');
    
    %сначала зададим дефолтный радиус, потом подгрузим если что (и степень тоже подгрузим)
    radius = 0.1;
    
    count_x = size(image,1); %N - количество пикселей в направлении оси ОХ
    count_y = size(image,2); %M - количество пикселей в направлении оси ОY

    A = 0.0;
    B = 0.0;
    C = 1.0;
    
    %рассматриваемый участок
    radius_look = linspace(-10^(-4), 10^(-4), count_points);
    count_radius = length(radius_look);
    
    %интенсивность в рассматриваемых точках z_look
    I_look = zeros(count_radius,1);
    %x_interpolate = radius_look(1):count_points:radius_look(count_points);
    %x_interpolate = abs(x_interpolate);
    
    %x = reshape(image.x, 1, countX * countY);
    
    %если не был запущен до этого трейсинг
    
    if(state == 0)   
        
        Traycing(h,eventdata);
        
    end 
    
    z_look = - str2num(get(distance2, 'String'));
    
    %intersection_parameters = Intersection_Parameters(intersection_parameters.x, intersection_parameters.y, intersection_parameters.z, count_x, count_y, A, B, C, z_look, intersection_parameters.I, intersection_parameters.phase, g_vector_lens);
    
    %если нужно будет, то и степень накрутим линзы
    R = ((intersection_parameters.x).^2 + (intersection_parameters.y).^2).^(1/2);
    
        for k = 1 : count_points - 1
            
            %для 
            if(radius_look(k) < 0)
                
                case1 = R >=  - radius_look(k+1);
                case2 = R <   - radius_look(k);
                
            else
                case1 = R <= radius_look(k+1);
                case2 = R >  radius_look(k);
            end
            
            case3 = case1 .* case2;
            
            %запишем матрицу в вектор, чтобы удобнее было с ним работать
            case3 = reshape(case3, 1, count_x * count_y);
            
            %расчёт полученной 
            I_curr = sum(case3, 'double');
            
            I_look(k) =  I_curr;
            
            disp(k)
        end
      
    %интерполяция результатов    
    %I_look = smooth(I_look);
       
    %вывод графика ФРТ на экран
    figure('Name','ФРТ');
    plot(radius_look, I_look, '-');
    grid on;
    title('Количество точек на отрезок');
    xlabel('Расстояние вдоль радиуса, м');
    
end

%удаление всех глобальных переменных
function [] = Deleting(h,eventdata)
    clear global all;  
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end