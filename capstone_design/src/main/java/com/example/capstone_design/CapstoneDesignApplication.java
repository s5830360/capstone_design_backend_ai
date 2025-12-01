package com.example.capstone_design;

import com.example.capstone_design.entity.UserAccount;
import com.example.capstone_design.repository.UserAccountRepository;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;

@SpringBootApplication
public class CapstoneDesignApplication {
	public static void main(String[] args) {
		SpringApplication.run(CapstoneDesignApplication.class, args);
	}

//	@Bean
//	CommandLineRunner init(UserAccountRepository repo) {
//		return args -> {
//			if (repo.count() == 0) {
//				repo.save(UserAccount.builder().username("tester").build());
//			}
//			System.out.println("[JPA OK] user_account count = " + repo.count());
//		};
//	}
}