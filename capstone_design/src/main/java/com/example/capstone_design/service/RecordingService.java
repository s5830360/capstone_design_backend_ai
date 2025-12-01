package com.example.capstone_design.service;

import com.example.capstone_design.dto.AnalysisResponse;
import com.example.capstone_design.entity.Recording;
import com.example.capstone_design.repository.RecordingRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.server.ResponseStatusException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Optional;
import java.util.UUID;

@Service
@RequiredArgsConstructor
public class RecordingService {
    private final RecordingRepository repo;
    private final AnalysisService analysisService;   // 🔹 FastAPI 연동

    // 저장 루트: application.yml에 없으면 유저 홈 아래 기본 경로 사용
    @Value("${storage.root:${user.home}/capstone_uploads}")
    private String storageRoot;

    private Path root() throws IOException {
        Path p = Path.of(storageRoot);
        if (!Files.exists(p)) Files.createDirectories(p);
        return p;
    }

    // 업로드 후 저장
    @Transactional
    public Recording save(
            String uploader,
            MultipartFile file,
            String description,
            Double latitude,
            Double longitude,
            String address,
            boolean emotionPublic
    ) {
        if (file == null || file.isEmpty()) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "file is required");
        }

        try {
            // 1. 파일 저장
            String original = file.getOriginalFilename();
            String ext = "";
            if (original != null && original.contains(".")) {
                ext = original.substring(original.lastIndexOf('.')); // .wav, .mp3 ...
            }
            String stored = UUID.randomUUID() + ext;

            Path target = root().resolve(stored);
            Files.copy(file.getInputStream(), target, StandardCopyOption.REPLACE_EXISTING);

            // 2. FastAPI 호출해서 감정 분석 수행
            AnalysisResponse analysis = analysisService.analyzeFile(file);

            // 3. Recording 엔티티 생성
            Recording r = Recording.builder()
                    .uploader(uploader)
                    .originalFilename(original == null ? "unknown" : original)
                    .storedFilename(stored)
                    .contentType(file.getContentType())
                    .size(file.getSize())
                    .description(description)
                    .latitude(latitude)
                    .longitude(longitude)
                    .address(address)
                    .emotion(analysis.getEmotion())          // FastAPI 결과
                    .confidence(analysis.getConfidence())    // FastAPI 결과
                    .emotionPublic(emotionPublic)
                    .build();

            return repo.save(r);
        } catch (IOException e) {
            throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, "file save failed");
        }
    }

    // ID로 녹음 데이터 조회
    @Transactional(readOnly = true)
    public Optional<Recording> findById(Long id) {
        return repo.findById(id);
    }
}
